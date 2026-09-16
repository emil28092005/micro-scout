"""Bounded source parsing and conservative static relationships."""

from __future__ import annotations

import ast
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from micro_scout.text import code_text, digest

EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".md": "markdown",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".micro-scout",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass(frozen=True)
class Symbol:
    id: str
    path: str
    name: str
    kind: str
    language: str
    start_line: int
    end_line: int
    content: str
    file_hash: str
    parent: str | None = None

    @property
    def model_text(self) -> str:
        return code_text(self.content, self.language)

    @property
    def lexical_text(self) -> str:
        return f"{self.path}\n{self.name}\n{self.content}"

    def to_dict(self) -> dict:
        return asdict(self)


def source_paths(root: Path, max_file_bytes: int = 1_000_000) -> list[Path]:
    """Honor gitignore when available; never traverse symlinks or hidden trees."""
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Repository root must be a directory")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        paths = [root / os.fsdecode(p) for p in result.stdout.split(b"\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        paths = []
        for directory, names, files in os.walk(root, followlinks=False):
            names[:] = [
                n
                for n in names
                if n not in EXCLUDED
                and not n.startswith(".")
                and not (Path(directory) / n).is_symlink()
            ]
            paths.extend(Path(directory) / name for name in files)
    selected = []
    for path in sorted(set(paths)):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() not in EXTENSIONS or path.is_symlink() or not path.is_file():
            continue
        if any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            continue
        if not path.resolve().is_relative_to(root) or path.stat().st_size > max_file_bytes:
            continue
        selected.append(path)
    return selected


def parse_source(relative: str, text: str, *, chunk_lines: int = 60) -> list[Symbol]:
    if chunk_lines < 1:
        raise ValueError("chunk_lines must be positive")
    language = EXTENSIONS.get(Path(relative).suffix.lower(), "text")
    lines = text.splitlines()
    file_hash = digest(text)
    symbols = []

    def add(name, kind, start, end, parent=None):
        content = "\n".join(lines[start - 1 : end])
        symbol_id = digest(f"{relative}:{name}:{start}:{end}:{file_hash}")[:24]
        symbol = Symbol(
            symbol_id, relative, name, kind, language, start, end, content, file_hash, parent
        )
        symbols.append(symbol)
        return symbol

    if language == "python":
        try:
            tree = ast.parse(text)

            def walk(node, prefix="", parent=None):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = f"{prefix}.{child.name}" if prefix else child.name
                        start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        symbol = add(
                            name,
                            "class" if isinstance(child, ast.ClassDef) else "function",
                            start,
                            child.end_lineno,
                            parent,
                        )
                        walk(child, name, symbol.id)
                    else:
                        walk(child, prefix, parent)

            walk(tree)
            covered = set()
            for symbol in symbols:
                covered.update(range(symbol.start_line, symbol.end_line + 1))
            # Module-level imports/constants are useful context too.
            start = None
            for number in range(1, len(lines) + 2):
                uncovered = number <= len(lines) and number not in covered
                if uncovered and start is None:
                    start = number
                if start is not None and (not uncovered or number - start >= chunk_lines):
                    if any(line.strip() for line in lines[start - 1 : number - 1]):
                        add(f"<module:{start}>", "module", start, number - 1)
                    start = number if uncovered else None
            return symbols
        except (SyntaxError, ValueError, RecursionError):
            symbols.clear()
    for start in range(0, len(lines), chunk_lines):
        end = min(start + chunk_lines, len(lines))
        if any(line.strip() for line in lines[start:end]):
            add(f"<chunk:{start + 1}>", "chunk", start + 1, end)
    return symbols


def build_edges(symbols: list[Symbol]) -> list[dict]:
    """Containment and unambiguous same-module bare-name calls only.

    Attribute calls, cross-module resolution, dynamic dispatch, and name shadowing
    are not claimed to be fully resolved. Calls are explicitly approximate.
    """
    names = defaultdict(list)
    for s in symbols:
        if s.kind == "function" and "." not in s.name:
            names[(s.path, s.name)].append(s.id)
    edges = set()
    for symbol in symbols:
        if symbol.parent:
            edges.add((symbol.parent, symbol.id, "contains"))
        if symbol.language != "python" or symbol.kind != "function":
            continue
        try:
            import textwrap

            tree = ast.parse(textwrap.dedent(symbol.content))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                targets = names[(symbol.path, node.func.id)]
                if len(targets) == 1 and targets[0] != symbol.id:
                    edges.add((symbol.id, targets[0], "possible_call"))
    return [{"source": s, "target": t, "kind": k} for s, t, k in sorted(edges)]
