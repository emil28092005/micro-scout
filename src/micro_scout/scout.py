"""Read-only retrieval harness with checked locations and bounded context."""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from micro_scout.index import Index
from micro_scout.lexical import BM25, reciprocal_rank_fusion, top_indices
from micro_scout.symbols import EXTENSIONS, Symbol
from micro_scout.text import digest

if TYPE_CHECKING:
    from micro_scout.encoder import Encoder


class StaleReferenceError(ValueError):
    """The indexed source no longer matches the file on disk."""


class Scout:
    def __init__(
        self, index: Index, encoder: Encoder | None = None, trace_path: Path | None = None
    ):
        self.index, self.encoder, self.trace_path = index, encoder, trace_path
        if encoder and index.metadata.get("encoder_fingerprint") != encoder.fingerprint:
            raise ValueError(
                "Model and index fingerprints differ; rebuild the index with this model"
            )
        self.lexical = BM25([s.lexical_text for s in index.symbols])
        self.recent_requests: OrderedDict[str, set[str]] = OrderedDict()
        self.neighbors = defaultdict(list)
        for edge in index.edges:
            self.neighbors[edge["source"]].append((edge["target"], edge["kind"], "outgoing"))
            self.neighbors[edge["target"]].append((edge["source"], edge["kind"], "incoming"))

    def _verified_content(self, symbol: Symbol, cache: dict[str, str]) -> str:
        if symbol.path not in cache:
            path = self.index.root / symbol.path
            relative = Path(symbol.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid path in index")
            try:
                if (
                    path.is_symlink()
                    or any(p.is_symlink() for p in path.parents if p != self.index.root.parent)
                    or not path.resolve(strict=True).is_relative_to(
                        self.index.root.resolve(strict=True)
                    )
                ):
                    raise ValueError("Source path escapes repository or uses a symlink")
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise StaleReferenceError(
                    f"Source unavailable: {symbol.path}; rebuild index"
                ) from exc
            if digest(text) != symbol.file_hash:
                raise StaleReferenceError(f"Source changed: {symbol.path}; rebuild index")
            cache[symbol.path] = text
        lines = cache[symbol.path].splitlines()
        if not 1 <= symbol.start_line <= symbol.end_line <= len(lines):
            raise ValueError("Invalid source range in index")
        content = "\n".join(lines[symbol.start_line - 1 : symbol.end_line])
        if content != symbol.content:
            raise ValueError("Indexed content does not match its source range")
        return content

    def _package(self, symbol: Symbol, cache: dict[str, str], max_chars: int) -> dict | None:
        content = self._verified_content(symbol, cache)
        lines, used = [], 0
        for line in content.splitlines():
            cost = len(line) + (1 if lines else 0)
            if used + cost > max_chars:
                break
            lines.append(line)
            used += cost
        if not lines:
            return None
        end = symbol.start_line + len(lines) - 1
        return {
            "id": symbol.id,
            "path": symbol.path,
            "name": symbol.name,
            "kind": symbol.kind,
            "language": symbol.language,
            "start_line": symbol.start_line,
            "end_line": end,
            "reference": f"{symbol.path}:{symbol.start_line}-{end}",
            "file_sha256": symbol.file_hash,
            "content": "\n".join(lines),
            "truncated": end < symbol.end_line,
            "verified": True,
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        max_chars: int = 12_000,
        mode: str = "hybrid",
        expand: bool = True,
        language: str | None = None,
        include_docs: bool = False,
    ) -> dict:
        started = time.perf_counter()
        if not query.strip() or len(query) > 8_000:
            raise ValueError("Query must contain 1–8000 characters")
        if not 1 <= top_k <= 50 or not 100 <= max_chars <= 100_000:
            raise ValueError("top_k must be 1–50 and max_chars 100–100000")
        if mode not in {"lexical", "dense", "hybrid"}:
            raise ValueError("mode must be lexical, dense, or hybrid")
        if language is not None and language not in set(EXTENSIONS.values()):
            raise ValueError("Unsupported language filter")
        if mode != "lexical" and (self.encoder is None or self.index.vectors is None):
            raise ValueError("Dense/hybrid search needs a model and dense index; use lexical mode")
        lexical = self.lexical.score(query)
        dense = None
        if mode != "lexical":
            dense = self.index.vectors @ self.encoder.encode([query], query=True)[0]
        eligible = np.array(
            [
                i
                for i, symbol in enumerate(self.index.symbols)
                if (language is None or symbol.language == language)
                and (include_docs or symbol.language != "markdown")
            ],
            dtype=np.int64,
        )
        scores = lexical[eligible] if mode == "lexical" else dense[eligible]
        if mode == "hybrid":
            scores = reciprocal_rank_fusion(lexical[eligible], dense[eligible])
        order = top_indices(scores, min(len(scores), max(100, top_k * 4)))
        cache, results, warnings, used_ids = {}, [], [], set()
        occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)

        def overlaps(symbol: Symbol) -> bool:
            return any(
                symbol.start_line <= end and symbol.end_line >= start
                for start, end in occupied[symbol.path]
            )

        def occupy(item: dict) -> None:
            occupied[item["path"]].append((item["start_line"], item["end_line"]))

        remaining = max_chars
        for position in order:
            if mode in {"lexical", "hybrid"} and scores[position] <= 0:
                continue
            i = int(eligible[position])
            symbol = self.index.symbols[i]
            if overlaps(symbol):
                continue
            try:
                result = self._package(symbol, cache, remaining)
            except (StaleReferenceError, ValueError) as exc:
                warnings.append(str(exc))
                continue
            if result is None:
                continue
            result.update(
                {
                    "score": float(scores[position]),
                    "retrieval": mode,
                    "bm25_score": float(lexical[i]),
                    "cosine_similarity": float(dense[i]) if dense is not None else None,
                }
            )
            results.append(result)
            occupy(result)
            used_ids.add(symbol.id)
            remaining -= len(result["content"])
            if len(results) >= top_k or remaining < 100:
                break
        # Neighbors use only remaining budget and do not displace ranked hits.
        related = []
        if expand and remaining >= 100:
            for result in results[:2]:
                for neighbor, kind, direction in self.neighbors[result["id"]]:
                    if neighbor in used_ids or overlaps(self.index.by_id[neighbor]):
                        continue
                    try:
                        item = self._package(
                            self.index.by_id[neighbor], cache, min(remaining, 2_000)
                        )
                    except (StaleReferenceError, ValueError) as exc:
                        warnings.append(str(exc))
                        continue
                    if item:
                        item.update(
                            {"relation": kind, "direction": direction, "from_id": result["id"]}
                        )
                        related.append(item)
                        occupy(item)
                        used_ids.add(neighbor)
                        remaining -= len(item["content"])
                    if len(related) >= 3 or remaining < 100:
                        break
                if len(related) >= 3 or remaining < 100:
                    break
        response = {
            "request_id": uuid.uuid4().hex,
            "query": query,
            "snapshot": self.index.metadata["snapshot"],
            "model_fingerprint": self.index.metadata.get("encoder_fingerprint"),
            "filters": {"language": language, "include_docs": include_docs},
            "results": results,
            "neighbors": related,
            "warnings": sorted(set(warnings)),
            "returned_chars": max_chars - remaining,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "freshness": (
                "returned files checked against indexed SHA-256; new files require reindexing"
            ),
        }
        self._trace(
            {
                "event": "search",
                "request_id": response["request_id"],
                "query": query,
                "snapshot": response["snapshot"],
                "model_fingerprint": response["model_fingerprint"],
                "mode": mode,
                "language": language,
                "include_docs": include_docs,
                "ids": [r["id"] for r in results],
                "neighbor_ids": [r["id"] for r in related],
                "top_k": top_k,
                "max_chars": max_chars,
                "expand": expand,
                "latency_ms": response["latency_ms"],
            }
        )
        self.recent_requests[response["request_id"]] = used_ids
        if len(self.recent_requests) > 1000:
            self.recent_requests.popitem(last=False)
        return response

    def read(self, symbol_id: str, max_chars: int = 12_000) -> dict:
        if not 100 <= max_chars <= 100_000:
            raise ValueError("max_chars must be 100–100000")
        if symbol_id not in self.index.by_id:
            raise ValueError("Unknown symbol ID; search the current index first")
        result = self._package(self.index.by_id[symbol_id], {}, max_chars)
        if result is None:
            raise ValueError("First source line exceeds the character budget")
        return result

    def feedback(self, request_id: str, useful_ids: list[str], outcome: str) -> dict:
        if self.trace_path is None:
            raise ValueError("Feedback requires --trace; weights are not changed online")
        if len(request_id) != 32 or any(c not in "0123456789abcdef" for c in request_id):
            raise ValueError("Invalid request_id")
        if outcome not in {"helpful", "unhelpful", "unknown"}:
            raise ValueError("outcome must be helpful, unhelpful, or unknown")
        if len(useful_ids) > 50 or any(i not in self.index.by_id for i in useful_ids):
            raise ValueError("Feedback contains invalid symbol IDs")
        if request_id not in self.recent_requests:
            raise ValueError("Unknown or expired request_id; use a search from this server session")
        if not set(useful_ids).issubset(self.recent_requests[request_id]):
            raise ValueError("Feedback IDs must have been returned by this search")
        self._trace(
            {
                "event": "feedback",
                "request_id": request_id,
                "useful_ids": useful_ids,
                "outcome": outcome,
                "snapshot": self.index.metadata["snapshot"],
            }
        )
        return {"recorded": True, "weights_updated": False}

    def _trace(self, event: dict) -> None:
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"schema_version": 1, "time": time.time(), **event}, ensure_ascii=False
                    )
                    + "\n"
                )
