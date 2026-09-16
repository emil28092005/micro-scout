"""Fixed, index-free keyword baseline for the development evaluation (no model)."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

from micro_scout.live_tools import LiveRepository

STOPWORDS = frozenset(
    [
        "find",
        "where",
        "which",
        "what",
        "when",
        "that",
        "this",
        "with",
        "from",
        "into",
        "implementation",
        "locate",
        "code",
        "function",
        "method",
        "returns",
        "return",
        "before",
        "after",
        "using",
        "used",
        "source",
        "current",
        "repository",
        "containing",
        "handles",
        "handling",
        "given",
        "does",
        "how",
        "the",
        "and",
        "for",
        "are",
    ]
)


def search_keywords(root: Path, query: str, *, max_chars: int = 6000) -> dict:
    """Up to twelve term searches, then three 25-line windows ranked by rare terms.

    Parameters are fixed for a simple control, not tuned against evaluation labels.
    This is not a simulation of a large model's adaptive grep strategy.
    """
    started = time.monotonic()
    repo = LiveRepository(root)
    terms = sorted({w.lower() for w in re.findall(r"[A-Za-z]{4,}", query)} - STOPWORDS)[:12]
    by_location, weights = {}, {}
    for term in terms:
        found = repo.grep(re.escape(term))["matches"]
        weights[term] = 1 / math.log2(2 + len(found))
        for match in found:
            by_location[(match["path"], match["line"])] = match
    matches = list(by_location.values())
    ranked = []
    for match in matches:
        start, end = max(1, match["line"] - 12), match["line"] + 12
        nearby = "\n".join(
            m["text"].lower()
            for m in matches
            if m["path"] == match["path"] and start <= m["line"] <= end
        )
        score = sum(weights[term] for term in terms if term in nearby)
        ranked.append((-score, match["path"], start, end))
    references, visited = [], []
    calls, errors, remaining = len(terms), 0, max_chars
    read_calls = 0
    for _, path, start, end in sorted(set(ranked)):
        if read_calls >= 3:
            break
        if any(p == path and start <= b and a <= end for p, a, b in visited):
            continue
        visited.append((path, start, end))
        calls += 1
        read_calls += 1
        read = repo.execute({"tool": "read", "path": path, "start_line": start, "end_line": end})
        if "error" in read:
            errors += 1
            continue
        if len(read["content"]) > remaining:
            continue
        remaining -= len(read["content"])
        references.append({k: read[k] for k in ("path", "start_line", "end_line")})
    results = repo.finish(references, max_chars)
    return {
        "query": query,
        "root": str(repo.root),
        "model": None,
        "status": "completed" if results else "abstained",
        "results": results,
        "warnings": [],
        "elapsed_seconds": time.monotonic() - started,
        "rounds": 0,
        "tool_calls": calls,
        "tool_errors": errors,
        "invalid_actions": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "returned_chars": sum(len(r["content"]) for r in results),
        "index_required": False,
    }


class KeywordBaseline:
    context = 0
    max_tokens = 0
    tokenizer = None

    @staticmethod
    def metadata():
        return {
            "backend": "keyword",
            "model": None,
            "max_query_terms": 12,
            "window_radius_lines": 12,
            "max_read_calls": 3,
            "ranking": "sum of 1/log2(2+returned_term_matches) for distinct terms in each window",
            "search": "one separate grep per term, each with the standard bounded output",
            "stopwords": sorted(STOPWORDS),
        }
