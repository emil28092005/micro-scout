"""Reproducible development evaluation of index-free localization, without a solver API."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

from micro_scout.agent import search_live
from micro_scout.io import atomic_json
from micro_scout.local_policy import OllamaPolicy


def score_locations(predictions: list[dict], targets: list[dict]) -> dict:
    expected = {
        (t["path"], line) for t in targets for line in range(t["start_line"], t["end_line"] + 1)
    }
    returned = {
        (p["path"], line) for p in predictions for line in range(p["start_line"], p["end_line"] + 1)
    }
    common = expected & returned
    hits = [
        len({(t["path"], line) for line in range(t["start_line"], t["end_line"] + 1)} & returned)
        >= min(3, t["end_line"] - t["start_line"] + 1)
        for t in targets
    ]
    precision = len(common) / len(returned) if returned else 0.0
    recall = len(common) / len(expected) if expected else 0.0
    return {
        "file_hit": bool({p["path"] for p in predictions} & {t["path"] for t in targets}),
        "target_hit": all(hits),
        "line_precision": precision,
        "line_recall": recall,
        "line_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "returned_lines": len(returned),
        "overlap_lines": len(common),
    }


class GpuSampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                memory, utilization = map(int, result.stdout.splitlines()[0].split(","))
                self.samples.append(
                    {
                        "unix_time": time.time(),
                        "memory_mib": memory,
                        "utilization_percent": utilization,
                    }
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(1)


def summarize(rows: list[dict]) -> dict:
    values = [r["result"]["elapsed_seconds"] for r in rows]
    return {
        "tasks": len(rows),
        "target_hit_rate": statistics.mean(r["scores"]["target_hit"] for r in rows),
        "file_hit_rate": statistics.mean(r["scores"]["file_hit"] for r in rows),
        "macro_line_precision": statistics.mean(r["scores"]["line_precision"] for r in rows),
        "macro_line_recall": statistics.mean(r["scores"]["line_recall"] for r in rows),
        "macro_line_f1": statistics.mean(r["scores"]["line_f1"] for r in rows),
        "latency_median_seconds": statistics.median(values),
        "latency_p95_seconds": statistics.quantiles(values, n=20, method="inclusive")[18]
        if len(values) > 1
        else values[0],
        "statuses": dict(Counter(r["result"]["status"] for r in rows)),
        "total_tool_errors": sum(r["result"]["tool_errors"] for r in rows),
        "total_invalid_actions": sum(r["result"]["invalid_actions"] for r in rows),
        "mean_rounds": statistics.mean(r["result"]["rounds"] for r in rows),
        "mean_tool_calls": statistics.mean(r["result"]["tool_calls"] for r in rows),
        "total_input_tokens": sum(r["result"]["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["result"]["output_tokens"] for r in rows),
        "mean_returned_chars": statistics.mean(r["result"]["returned_chars"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("evals/live-search-v1.json"))
    parser.add_argument("--repos", type=Path, default=Path("data/search-eval"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openbmb/minicpm5:q4_K_M")
    parser.add_argument(
        "--backend", choices=["ollama", "transformers", "keyword"], default="ollama"
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    freeze = args.output / "experiment.json"
    if freeze.exists():
        parser.error("Output already contains a frozen experiment; choose a new output directory")
    for name, metadata in suite["repositories"].items():
        root = args.repos / name
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if commit != metadata["commit"]:
            parser.error(f"Repository revision mismatch: {name}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
            parser.error(f"Evaluation repository is not clean: {name}")
    for task in suite["tasks"]:
        for target in task["targets"]:
            path = args.repos / task["repository"] / target["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != target["sha256"]:
                parser.error(f"Source changed: {path}")
    if args.adapter and args.backend != "transformers":
        parser.error("--adapter requires --backend transformers")
    if args.backend == "keyword":
        from micro_scout.keyword_baseline import KeywordBaseline

        policy = KeywordBaseline()
    elif args.backend == "transformers":
        from micro_scout.transformers_policy import TransformersPolicy

        policy = TransformersPolicy(adapter=args.adapter)
    else:
        policy = OllamaPolicy(args.model)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    atomic_json(
        freeze,
        {
            "suite": suite,
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
            "model": policy.metadata(),
            "source_sha256": source_hashes,
            "max_rounds": args.max_rounds if args.backend != "keyword" else None,
            "max_chars": 6000,
            "timeout_seconds": 90 if args.backend != "keyword" else None,
            "context": policy.context,
            "max_generation_tokens": policy.max_tokens,
            "tokenizer": ("pinned_hf" if policy.tokenizer else "utf8_upper_bound")
            if args.backend != "keyword"
            else None,
            "temperature": 0,
            "seed": 42,
            "scope": "Single-function localization; no large-model baseline; "
            "public repositories may have appeared in the base model's pretraining.",
        },
    )
    # Explicit warm-up excludes runtime initialization from the measured task distribution.
    if args.backend == "ollama":
        warm = policy.request(
            "/api/generate",
            {
                "model": policy.model,
                "prompt": "Hello",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_ctx": policy.context, "num_predict": 1, "temperature": 0},
            },
        )
    elif args.backend == "transformers":
        from micro_scout.native_protocol import SYSTEM_PROMPT

        warm = policy.generate(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "List source files in the current repository."},
            ]
        )
    else:
        warm = {"skipped": "Keyword baseline has no model"}
    atomic_json(args.output / "warmup.json", warm)
    sampler = GpuSampler()
    if args.backend != "keyword":
        sampler.thread.start()
    rows = []
    try:
        for task in suite["tasks"]:
            if args.backend == "keyword":
                from micro_scout.keyword_baseline import search_keywords

                result = search_keywords(args.repos / task["repository"], task["query"])
            else:
                result = search_live(
                    args.repos / task["repository"],
                    task["query"],
                    policy,
                    max_rounds=args.max_rounds,
                    trace=args.output / f"{task['id']}.trace.json",
                )
            row = {
                "id": task["id"],
                "repository": task["repository"],
                "result": result,
                "scores": score_locations(result["results"], task["targets"]),
            }
            rows.append(row)
            atomic_json(args.output / "results.json", rows)
            print(
                json.dumps(
                    {
                        "id": task["id"],
                        "status": result["status"],
                        "hit": row["scores"]["target_hit"],
                        "seconds": round(result["elapsed_seconds"], 2),
                    }
                ),
                flush=True,
            )
    finally:
        sampler.stop.set()
        if sampler.thread.is_alive():
            sampler.thread.join(timeout=3)
        atomic_json(args.output / "gpu-samples.json", sampler.samples)
    summary = summarize(rows)
    summary["per_repository"] = {
        name: summarize([r for r in rows if r["repository"] == name])
        for name in suite["repositories"]
    }
    summary["sampled_peak_gpu_memory_mib"] = max(
        (s["memory_mib"] for s in sampler.samples),
        default=None,
    )
    summary["gpu_sampling_interval_seconds"] = 1
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
