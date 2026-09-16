"""Deterministic retrieval metrics with explicit single-positive assumptions."""

from __future__ import annotations

import numpy as np


def ranks_from_scores(scores: np.ndarray, positives: np.ndarray | None = None) -> np.ndarray:
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite query-by-candidate matrix")
    if positives is None:
        positives = np.arange(len(scores))
    if (
        len(positives) != len(scores)
        or np.any(positives < 0)
        or np.any(positives >= scores.shape[1])
    ):
        raise ValueError("Invalid positive candidate indices")
    # Stable index tie-breaking is shared by evaluation and serving.
    target = scores[np.arange(len(scores)), positives, None]
    better = scores > target
    tied_before = (scores == target) & (np.arange(scores.shape[1])[None, :] < positives[:, None])
    return 1 + np.sum(better | tied_before, axis=1)


def retrieval_metrics(ranks: np.ndarray) -> dict[str, float]:
    if not len(ranks) or np.any(ranks < 1):
        raise ValueError("Need at least one positive rank")
    return {
        "mrr": float(np.mean(1.0 / ranks)),
        "mrr_at_10": float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0))),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
    }
