"""Compare frozen models on identical candidates, with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from micro_scout.encoder import BASE_MODEL, Encoder
from micro_scout.io import atomic_json, read_jsonl, write_jsonl
from micro_scout.lexical import BM25, reciprocal_rank_fusion
from micro_scout.metrics import ranks_from_scores, retrieval_metrics


def paired_mrr_interval(
    a: np.ndarray, b: np.ndarray, repositories: list[str], seed: int = 17
) -> dict:
    """Resample whole repositories to retain correlation among their functions."""
    if len(a) != len(b) or len(a) != len(repositories) or not len(a):
        raise ValueError("Paired ranks and repository labels must have the same nonzero length")
    differences = 1 / a - 1 / b
    rng = np.random.default_rng(seed)
    groups = {repo: [] for repo in sorted(set(repositories))}
    for repo, difference in zip(repositories, differences, strict=True):
        groups[repo].append(difference)
    sums = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    choices = rng.integers(0, len(groups), size=(2000, len(groups)))
    samples = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
    return {
        "delta": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "method": "paired repository-cluster bootstrap, 2000 resamples",
        "repository_clusters": len(groups),
    }


def evaluate(
    data: Path, model: str, output: Path, split: str, device: str, limit: int | None = None
) -> dict:
    rows = read_jsonl(data / f"{split}.jsonl")
    dataset_hash = hashlib.sha256((data / f"{split}.jsonl").read_bytes()).hexdigest()
    manifest = json.loads((data / "manifest.json").read_text())
    if dataset_hash != manifest["splits"][split]["prepared_sha256"]:
        raise ValueError(f"Dataset checksum mismatch for {split}")
    if limit is not None:
        if limit < 2:
            raise ValueError("Evaluation needs at least two candidates")
        rows = rows[:limit]
    if len(rows) < 2:
        raise ValueError("Evaluation needs at least two candidates")
    output.mkdir(parents=True, exist_ok=True)
    queries, codes = [r["query"] for r in rows], [r["code"] for r in rows]
    started = time.perf_counter()
    bm25 = BM25(codes)
    lexical = np.stack([bm25.score(q) for q in queries])
    ranks = {"bm25": ranks_from_scores(lexical)}
    report = {
        "split": split,
        "queries": len(rows),
        "candidates_per_query": len(rows),
        "repositories": len({r["repo"] for r in rows}),
        "dataset_sha256": dataset_hash,
        "evaluation_ids_sha256": hashlib.sha256(
            json.dumps([r["id"] for r in rows]).encode()
        ).hexdigest(),
        "protocol": (
            "one labeled positive per query; all split snippets are candidates; docstrings removed"
        ),
        "limitations": [
            "Other semantically correct snippets may be counted as negatives",
            "This is function retrieval, not a bug-fixing or multi-file context benchmark",
            "No downstream Astra task-success evaluation has been run",
            "Base-model pretraining contamination cannot be excluded",
        ],
        "models": {},
        "metrics": {},
    }
    for label, name in (("pretrained", BASE_MODEL), ("trained", model)):
        print(json.dumps({"event": "encoding", "model": label, "examples": len(rows)}), flush=True)
        encoder = Encoder(name, device=device)
        t0 = time.perf_counter()
        code_vectors = encoder.encode(codes)
        query_vectors = encoder.encode(queries, query=True)
        dense = query_vectors @ code_vectors.T
        ranks[f"{label}_dense"] = ranks_from_scores(dense)
        hybrid = np.stack([reciprocal_rank_fusion(lexical[i], dense[i]) for i in range(len(rows))])
        ranks[f"{label}_hybrid"] = ranks_from_scores(hybrid)
        report["models"][label] = {
            "fingerprint": encoder.fingerprint,
            "parameters": encoder.parameter_count,
            "config": encoder.config,
            "embedding_seconds": time.perf_counter() - t0,
        }
        del encoder, code_vectors, query_vectors, dense, hybrid
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
    for name, values in ranks.items():
        report["metrics"][name] = retrieval_metrics(values)
        groups = {}
        for row, rank in zip(rows, values, strict=True):
            groups.setdefault(row["repo"], []).append(1 / rank)
        report["metrics"][name]["macro_repository_mrr"] = float(
            np.mean([np.mean(v) for v in groups.values()])
        )
    report["trained_vs_pretrained_dense_mrr"] = paired_mrr_interval(
        ranks["trained_dense"], ranks["pretrained_dense"], [r["repo"] for r in rows]
    )
    report["trained_hybrid_vs_bm25_mrr"] = paired_mrr_interval(
        ranks["trained_hybrid"], ranks["bm25"], [r["repo"] for r in rows]
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(output / f"{split}-metrics.json", report)
    write_jsonl(
        output / f"{split}-ranks.jsonl",
        [
            {"id": row["id"], "repo": row["repo"], **{name: int(v[i]) for name, v in ranks.items()}}
            for i, row in enumerate(rows)
        ],
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csn-python-v1"))
    parser.add_argument("--model", default="runs/minilm-v1/best")
    parser.add_argument("--output", type=Path, default=Path("runs/minilm-v1/evaluation"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate(args.data, args.model, args.output, args.split, args.device, args.limit)


if __name__ == "__main__":
    main()
