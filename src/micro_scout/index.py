"""Atomic SQLite snapshots with optional reusable dense embeddings."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from micro_scout.symbols import Symbol, build_edges, parse_source, read_source, source_paths
from micro_scout.text import digest

if TYPE_CHECKING:
    from micro_scout.encoder import Encoder

SCHEMA_VERSION = 1


class Index:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            self.metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            self.metadata = {k: json.loads(v) for k, v in self.metadata.items()}
            if self.metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Unsupported index schema; rebuild the index")
            self.symbols = []
            vectors = []
            for payload, vector in connection.execute(
                "SELECT payload, vector FROM symbols ORDER BY ordinal"
            ):
                self.symbols.append(Symbol(**json.loads(payload)))
                if vector is not None:
                    vectors.append(np.frombuffer(vector, dtype="<f4"))
            self.edges = [
                json.loads(row[0]) for row in connection.execute("SELECT payload FROM edges")
            ]
            self.files = dict(connection.execute("SELECT path, file_hash FROM files"))
        finally:
            connection.close()
        if len({s.id for s in self.symbols}) != len(self.symbols):
            raise ValueError("Index contains duplicate symbol IDs")
        self.root = Path(self.metadata["root"])
        self.vectors = np.stack(vectors) if vectors else None
        if self.vectors is not None:
            expected = (len(self.symbols), self.metadata["dimension"])
            if self.vectors.shape != expected or not np.isfinite(self.vectors).all():
                raise ValueError("Corrupt embedding matrix; rebuild the index")
        elif self.metadata.get("encoder_fingerprint") and self.symbols:
            raise ValueError("Dense index has missing embeddings")
        self.by_id = {s.id: s for s in self.symbols}


def build_index(
    root: Path, output: Path, encoder: Encoder | None = None, *, max_symbols: int = 50_000
) -> dict:
    started = time.monotonic()
    root = root.resolve(strict=True)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    symbols, files, warnings = [], {}, []
    for path in source_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            text = read_source(path)
        except (ValueError, OSError) as exc:
            warnings.append({"path": relative, "reason": type(exc).__name__})
            continue
        files[relative] = digest(text)
        symbols.extend(parse_source(relative, text))
        if len(symbols) > max_symbols:
            raise ValueError(
                f"Index exceeds {max_symbols} symbols; choose a smaller repository root"
            )
    if not symbols:
        raise ValueError("No supported source files found")
    old_vectors = {}
    if encoder and output.is_file():
        previous = Index(output)
        if (
            previous.metadata.get("encoder_fingerprint") == encoder.fingerprint
            and previous.vectors is not None
        ):
            old_vectors = {
                digest(s.model_text): vector
                for s, vector in zip(previous.symbols, previous.vectors, strict=True)
            }
    vectors = None
    reused = 0
    if encoder:
        vectors = np.empty((len(symbols), encoder.dimension), dtype=np.float32)
        missing, texts = [], []
        for i, symbol in enumerate(symbols):
            text = symbol.model_text
            cached = old_vectors.get(digest(text))
            if cached is None:
                missing.append(i)
                texts.append(text)
            else:
                vectors[i] = cached
                reused += 1
        if texts:
            vectors[missing] = encoder.encode(texts)
    edges = build_edges(symbols)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "snapshot": digest(json.dumps(files, sort_keys=True)),
        "created_at_unix": time.time(),
        "files": len(files),
        "symbols": len(symbols),
        "edges": len(edges),
        "encoder_fingerprint": encoder.fingerprint if encoder else None,
        "dimension": encoder.dimension if encoder else None,
        "reused_embeddings": reused,
        "warnings": warnings,
        "build_seconds": time.monotonic() - started,
    }
    fd, temporary = tempfile.mkstemp(prefix=".scout-index-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE files (path TEXT PRIMARY KEY, file_hash TEXT NOT NULL);
                CREATE TABLE symbols (
                    ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL, vector BLOB
                );
                CREATE TABLE edges (payload TEXT NOT NULL);
            """)
            with connection:
                connection.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    [(k, json.dumps(v)) for k, v in metadata.items()],
                )
                connection.executemany("INSERT INTO files VALUES (?, ?)", files.items())
                connection.executemany(
                    "INSERT INTO symbols VALUES (?, ?, ?)",
                    [
                        (
                            i,
                            json.dumps(s.to_dict()),
                            vectors[i].astype("<f4").tobytes() if vectors is not None else None,
                        )
                        for i, s in enumerate(symbols)
                    ],
                )
                connection.executemany(
                    "INSERT INTO edges VALUES (?)", [(json.dumps(e),) for e in edges]
                )
        finally:
            connection.close()
        # Readers see the old complete snapshot or the new complete snapshot.
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata
