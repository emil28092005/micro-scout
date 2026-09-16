"""Shared training and serving text normalization. No code is executed."""

from __future__ import annotations

import ast
import hashlib
import io
import re
import textwrap
import tokenize


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_python_documentation(code: str) -> str:
    """Remove docstrings and comments without changing executable string literals.

    AST parsing is intentionally strict in the dataset pipeline. Serving may catch
    SyntaxError and use raw text for incomplete files. Byte offsets in the Python
    AST are handled through UTF-8 encoded lines, including non-ASCII identifiers.
    """
    code = textwrap.dedent(code)
    tree = ast.parse(code)
    lines = code.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                spans.append(
                    (
                        offsets[first.lineno - 1] + first.col_offset,
                        offsets[first.end_lineno - 1] + first.end_col_offset,
                    )
                )
    raw = bytearray(code.encode("utf-8"))
    for start, end in spans:
        for i in range(start, end):
            if raw[i] not in (10, 13):
                raw[i] = 32
    stripped = raw.decode("utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(stripped).readline)
    return tokenize.untokenize(t for t in tokens if t.type != tokenize.COMMENT).strip()


def code_fingerprints(code: str) -> tuple[str, str]:
    """Token hash plus an identifier/literal-normalized clone heuristic.

    This is a conservative near-clone filter, not proof of no training leakage.
    Keywords and operator structure are retained; formatting/comments are not.
    """
    import keyword

    exact, structural = [], []
    ignore = {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}
    for t in tokenize.generate_tokens(io.StringIO(code).readline):
        if t.type in ignore or t.type == tokenize.COMMENT:
            continue
        value = "" if t.type in (tokenize.INDENT, tokenize.DEDENT) else t.string
        exact.append((t.type, value))
        if t.type == tokenize.NAME and not keyword.iskeyword(value):
            value = "NAME"
        elif t.type in (tokenize.NUMBER, tokenize.STRING):
            value = "LITERAL"
        structural.append((t.type, value))
    return digest(repr(exact)), digest(repr(structural))


def code_text(code: str, language: str = "python") -> str:
    if language == "python":
        try:
            return strip_python_documentation(code)
        except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
            pass
    return code


def lexical_tokens(text: str) -> list[str]:
    """Keep exact names and add snake_case/camelCase components."""
    result = []
    for word in re.findall(r"[^\W_]+(?:_[^\W_]+)*", text, re.UNICODE):
        result.append(word.lower())
        pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").split()
        if len(pieces) > 1:
            result.extend(piece.lower() for piece in pieces)
    return result
