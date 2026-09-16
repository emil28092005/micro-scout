"""Read-only, bounded filesystem tools for search without a persistent index."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import stat
import subprocess
import time
from pathlib import Path


class LiveRepository:
    """Each instance belongs to one search; observations are never shared across queries."""

    def __init__(self, root: Path, *, timeout: float = 3.0):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Repository root must be a directory")
        self.rg = shutil.which("rg")
        if not self.rg:
            raise ValueError("Index-free search requires ripgrep (rg) on PATH")
        self.timeout = timeout
        self.observed: dict[str, tuple[str, set[int]]] = {}

    @staticmethod
    def _relative(path: str) -> Path:
        if not isinstance(path, str) or not path or len(path) > 1024 or "\x00" in path:
            raise ValueError("Invalid relative path")
        rel = Path(path)
        if rel.is_absolute() or any(p == ".." or p.startswith(".") for p in rel.parts):
            raise ValueError("Paths must stay inside the repository; hidden paths are excluded")
        return rel

    def _source(self, path: str) -> tuple[list[str], str]:
        """Use openat with O_NOFOLLOW, including parents, to reject symlink races."""
        rel = self._relative(path)
        if not rel.parts:
            raise ValueError("Expected a file")
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in rel.parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            descriptor = os.open(
                rel.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            with os.fdopen(descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Expected a regular source file")
                raw = stream.read(1_000_001)
        finally:
            os.close(directory)
        if len(raw) > 1_000_000 or b"\x00" in raw:
            raise ValueError("File is binary or exceeds 1 MB")
        return raw.decode("utf-8").splitlines(), hashlib.sha256(raw).hexdigest()

    def _run(self, args: list[str]) -> tuple[bytes, bool]:
        command = [self.rg, "--no-config", "--color=never", *args]
        for excluded in (".git", ".venv", "venv", "node_modules", "vendor", "dist", "build"):
            command.extend(["--glob", f"!**/{excluded}/**"])
        # '--' and '.' are added by the caller only after all option arguments.
        process = subprocess.Popen(
            command + ["--", "."],
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "RIPGREP_CONFIG_PATH": ""},
        )
        output, errors = bytearray(), bytearray()
        deadline = time.monotonic() + self.timeout
        limited = False
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, output)
                selector.register(process.stderr, selectors.EVENT_READ, errors)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        limited = True
                        break
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            key.data.extend(chunk)
                    if len(output) + len(errors) > 262_144:
                        limited = True
                        break
            if limited:
                process.kill()
            process.wait(timeout=1)
            if not limited and process.returncode not in (0, 1):
                raise ValueError(errors[:1000].decode("utf-8", errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
        return bytes(output), limited

    def _glob(self, glob: str) -> list[str]:
        if not isinstance(glob, str) or len(glob) > 256 or "\x00" in glob:
            raise ValueError("Invalid glob")
        if glob.startswith(("/", "!")) or ".." in Path(glob).parts:
            raise ValueError("Glob must be a relative include pattern")
        if glob.endswith("/"):
            glob += "**"
        return ["--glob", glob] if glob else []

    def files(self, glob: str = "") -> dict:
        raw, limited = self._run(["--files", "--null", *self._glob(glob)])
        allowed = None
        if glob:
            visible, inventory_limited = self._run(["--files", "--null"])
            allowed = {os.fsdecode(p).removeprefix("./") for p in visible.split(b"\0")[:-1]}
            limited |= inventory_limited
        paths = []
        # A capped subprocess may end in a partial path; discard it.
        for entry in raw.split(b"\0")[:-1]:
            path = os.fsdecode(entry).removeprefix("./")
            if allowed is not None and path not in allowed:
                continue
            try:
                self._relative(path)
            except ValueError:
                continue
            paths.append(path)
        ordered = sorted(paths, key=lambda p: (not p.startswith(("src/", "lib/")), p))
        return {"files": ordered[:100], "truncated": limited or len(paths) > 100}

    def grep(self, pattern: str, glob: str = "") -> dict:
        if not isinstance(pattern, str) or not 1 <= len(pattern) <= 256 or "\x00" in pattern:
            raise ValueError("Pattern must contain 1–256 characters")
        if "\n" in pattern or "\r" in pattern:
            raise ValueError("Only single-line regex patterns are supported")
        raw, limited = self._run(
            [
                "--json",
                "--ignore-case",
                "--max-count",
                "8",
                "--max-filesize",
                "1M",
                *self._glob(glob),
                "-e",
                pattern,
            ]
        )
        matches = []
        allowed = None
        if glob:
            # Positive rg globs override gitignore. Filter against an unmodified
            # inventory so model-generated globs cannot expose ignored matches.
            visible, inventory_limited = self._run(["--files", "--null"])
            allowed = {os.fsdecode(p).removeprefix("./") for p in visible.split(b"\0")[:-1]}
            limited |= inventory_limited
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get("type") != "match":
                continue
            item = event["data"]
            path = item["path"].get("text", "").removeprefix("./")
            if allowed is not None and path not in allowed:
                continue
            try:
                self._relative(path)
            except ValueError:
                continue
            matches.append(
                {
                    "path": path,
                    "line": item["line_number"],
                    "text": item["lines"].get("text", "")[:220].rstrip(),
                }
            )
        matches.sort(
            key=lambda r: (not r["path"].startswith(("src/", "lib/")), r["path"], r["line"])
        )
        return {
            "matches": matches[:30],
            "truncated": limited or len(matches) > 30,
            "per_file_match_limit": 8,
        }

    def read(self, path: str, start_line: int, end_line: int) -> dict:
        if type(start_line) is not int or type(end_line) is not int:
            raise ValueError("Line numbers must be integers")
        if start_line < 1 or end_line < start_line or end_line - start_line >= 120:
            raise ValueError("Read 1–120 lines using one-based inclusive ranges")
        lines, sha = self._source(path)
        if start_line > len(lines):
            raise ValueError(f"Start exceeds file length ({len(lines)} lines)")
        end_line = min(end_line, len(lines))
        selected, chars = [], 0
        for number in range(start_line, end_line + 1):
            text = lines[number - 1]
            if chars + len(text) + 16 > 8000:
                break
            selected.append(f"{number}: {text}")
            chars += len(text) + 16
        if not selected:
            raise ValueError("Source line exceeds the read output budget")
        actual_end = start_line + len(selected) - 1
        previous_sha, seen = self.observed.get(path, (sha, set()))
        seen = seen if previous_sha == sha else set()
        seen.update(range(start_line, actual_end + 1))
        self.observed[path] = (sha, seen)
        return {
            "path": path,
            "start_line": start_line,
            "end_line": actual_end,
            "file_lines": len(lines),
            "sha256": sha,
            "content": "\n".join(selected),
            "truncated": actual_end < end_line,
        }

    def finish(self, references: list[dict], max_chars: int) -> list[dict]:
        if not isinstance(references, list) or len(references) > 5:
            raise ValueError("Return at most five references")
        results, emitted = [], set()
        remaining = max_chars
        for ref in references:
            if not isinstance(ref, dict) or set(ref) != {"path", "start_line", "end_line"}:
                raise ValueError("Each reference needs exactly path, start_line, end_line")
            if not isinstance(ref["path"], str):
                raise ValueError("Reference path must be a string")
            path, start, end = ref["path"], ref["start_line"], ref["end_line"]
            if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                raise ValueError("Invalid final line range")
            if end - start >= 120:
                raise ValueError("A final range may contain at most 120 lines")
            old_sha, seen = self.observed.get(path, (None, set()))
            if not set(range(start, end + 1)).issubset(seen):
                raise ValueError(
                    f"Read the complete range before returning it: {path}:{start}-{end}"
                )
            lines, sha = self._source(path)
            if sha != old_sha:
                raise ValueError(f"Source changed during search; read again: {path}")
            if any((path, n) in emitted for n in range(start, end + 1)):
                continue
            content = "\n".join(lines[start - 1 : end])
            if len(content) > remaining:
                raise ValueError("Final source exceeds max_chars; choose shorter ranges")
            emitted.update((path, n) for n in range(start, end + 1))
            remaining -= len(content)
            results.append(
                {
                    **ref,
                    "reference": f"{path}:{start}-{end}",
                    "content": content,
                    "sha256": sha,
                    "verified": True,
                }
            )
        return results

    def execute(self, call: dict) -> dict:
        allowed = {"files": self.files, "grep": self.grep, "read": self.read}
        try:
            if not isinstance(call, dict):
                raise ValueError("Tool call must be an object")
            name = call["tool"]
            if name not in allowed:
                raise ValueError("Unknown tool")
            return allowed[name](**{k: v for k, v in call.items() if k != "tool"})
        except (ValueError, OSError, TypeError, KeyError) as exc:
            return {"error": str(exc)[:1000]}
