"""A compact BM25 baseline and deterministic reciprocal-rank fusion."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from micro_scout.text import lexical_tokens


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.size = len(texts)
        self.k1, self.b = k1, b
        postings = defaultdict(list)
        lengths = []
        for i, text in enumerate(texts):
            counts = Counter(lexical_tokens(text))
            lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                postings[token].append((i, frequency))
        lengths = np.array(lengths, dtype=np.float32)
        average = float(lengths.mean()) if len(lengths) else 1.0
        self.norm = k1 * (1 - b + b * lengths / max(average, 1.0))
        self.postings = {}
        for token, items in postings.items():
            indices, counts = np.array(items, dtype=np.int64).T
            idf = np.log(1 + (self.size - len(indices) + 0.5) / (len(indices) + 0.5))
            self.postings[token] = (
                indices,
                idf * counts * (k1 + 1) / (counts + self.norm[indices]),
            )

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.size, dtype=np.float32)
        for token in set(lexical_tokens(query)):
            if token in self.postings:
                indices, weights = self.postings[token]
                scores[indices] += weights
        return scores


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k < 0:
        raise ValueError("k must not be negative")
    # Full stable sort is fast for the intended small-repository MVP.
    return np.argsort(-scores, kind="stable")[:k]


def reciprocal_rank_fusion(
    lexical: np.ndarray, dense: np.ndarray, *, weight: float = 0.5, limit: int = 100, k: int = 60
) -> np.ndarray:
    if lexical.shape != dense.shape or not 0 <= weight <= 1 or limit < 1 or k < 1:
        raise ValueError("Invalid fusion settings")
    result = np.zeros_like(dense, dtype=np.float32)
    for scores, contribution, positive_only in (
        (lexical, 1 - weight, True),
        (dense, weight, False),
    ):
        order = top_indices(scores, min(limit, len(scores)))
        if positive_only:
            order = order[scores[order] > 0]
        result[order] += contribution / (k + np.arange(1, len(order) + 1))
    return result
