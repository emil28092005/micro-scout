"""Pinned CodeSearchNet preparation with auditable, repository-disjoint splits."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import tempfile
import tokenize
from collections import Counter
from pathlib import Path

from micro_scout.io import atomic_json, write_jsonl
from micro_scout.text import code_fingerprints, digest, strip_python_documentation

DATASET = "code-search-net/code_search_net"
REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"
SOURCE_SHA256 = {
    "train": "ad9e3a4ab10c2c1d8926d2b26ca2bfcc3aadda1477ba29a933391f93806b9fed",
    "validation": "22eaacb46ed7e74d582409b85692ef63f5a43e99f9395c2eb736b5c8451422bb",
    "test": "3167e79ee7f081d825bf97b96d3a6b2d96428b00f6a98125be943384d8afae5f",
}


def normalize_row(raw: dict) -> dict | None:
    """Use only code as model input and the first documentation paragraph as query."""
    query = str(raw.get("func_documentation_string", raw.get("docstring", ""))).strip()
    query = re.split(r"\n\s*\n|\n\s*(?:Args:|Parameters|:param|Returns:)", query)[0]
    query = " ".join(query.split())
    if not 4 <= len(query.split()) <= 80 or len(query) > 700:
        return None
    repo = raw.get("repository_name", raw.get("repo", ""))
    code = raw.get("func_code_string", raw.get("code", ""))
    path = raw.get("func_path_in_repository", raw.get("path", ""))
    url = raw.get("func_code_url", raw.get("url", ""))
    if not repo or not path or not url.startswith("https://github.com/"):
        return None
    if not 80 <= len(code) <= 12_000 or len(code.splitlines()) > 140:
        return None
    try:
        code = strip_python_documentation(code)
        exact, structural = code_fingerprints(code)
    except (SyntaxError, ValueError, tokenize.TokenError, IndentationError):
        return None
    if len(code) < 60 or len(code.split()) > 600 or query.lower() in code.lower():
        return None
    return {
        "id": digest(url + "\n" + exact)[:24],
        "query": query,
        "code": code,
        "repo": repo.lower(),
        "path": path,
        "url": url,
        "language": "python",
        "code_hash": exact,
        "structural_hash": structural,
        "query_hash": digest(query.lower()),
        "source": DATASET,
        "source_revision": REVISION,
    }


def prepare(source: Path, output: Path, sizes: dict[str, int], seed: int = 17) -> dict:
    """Publish a complete dataset only after all splits pass the overlap audit."""
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        prepared = Path(staging) / "dataset"
        result = _prepare(source, prepared, sizes, seed)
        os.replace(prepared, output)
    return result


def _prepare(source: Path, output: Path, sizes: dict[str, int], seed: int = 17) -> dict:
    import pyarrow.parquet as pq

    if any(n < 1 for n in sizes.values()):
        raise ValueError("Split sizes must be positive")
    if output.exists() and any(output.glob("*.jsonl")):
        raise ValueError("Output already contains a dataset; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    seen_global: dict[str, set[str]] = {
        key: set() for key in ("repo", "code_hash", "structural_hash", "query_hash")
    }
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "seed": seed,
        "selection": "seeded hash priority, maximum 200 samples per repository",
        "normalization": "remove Python docstrings/comments; first documentation paragraph",
        "deduplication": "repository, exact token hash, normalized token hash, exact query hash",
        "splits": {},
    }
    # Freeze evaluation first, then remove overlap from development and training.
    for split in ("test", "validation", "train"):
        n = sizes[split]
        source_file = source / f"{split}.parquet"
        sha = hashlib.sha256()
        with source_file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != SOURCE_SHA256[split]:
            raise ValueError(f"Source checksum mismatch for {split}; download the pinned dataset")
        counts: Counter = Counter()
        seen_local: set[str] = set()
        heap = []
        for batch in pq.ParquetFile(source_file).iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                counts["source_rows"] += 1
                row = normalize_row(raw)
                if row is None:
                    counts["filtered_quality"] += 1
                    continue
                if row["code_hash"] in seen_local:
                    counts["duplicate_code_in_split"] += 1
                    continue
                seen_local.add(row["code_hash"])
                overlap = next((k for k, seen in seen_global.items() if row[k] in seen), None)
                if overlap:
                    counts[f"overlap_{overlap}"] += 1
                    continue
                priority = int(digest(f"{seed}:{row['id']}"), 16)
                item = (-priority, row["id"], row)
                if len(heap) < n * 4:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if counts["source_rows"] % 51200 == 0:
                print(json.dumps({"split": split, **counts}), flush=True)
        rows, repos, fingerprints, queries = [], Counter(), set(), set()
        for _, _, row in sorted(heap, reverse=True):
            if repos[row["repo"]] >= 200:
                continue
            if row["structural_hash"] in fingerprints or row["query_hash"] in queries:
                continue
            repos[row["repo"]] += 1
            fingerprints.add(row["structural_hash"])
            queries.add(row["query_hash"])
            row["split"] = split
            rows.append(row)
            if len(rows) == n:
                break
        if len(rows) < min(n, 100):
            raise ValueError(f"Insufficient clean {split} examples: {len(rows)}")
        for row in rows:
            for key, seen in seen_global.items():
                seen.add(row[key])
        write_jsonl(output / f"{split}.jsonl", rows)
        report["splits"][split] = {
            **counts,
            "selected": len(rows),
            "repositories": len(repos),
            "max_repository_examples": max(repos.values()),
            "source_sha256": sha.hexdigest(),
            "prepared_sha256": hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest(),
        }
        print(json.dumps({"split": split, **report["splits"][split]}), flush=True)
    report["overlap_audit"] = audit_splits(output)
    atomic_json(output / "manifest.json", report)
    return report


def audit_splits(path: Path) -> dict:
    from micro_scout.io import read_jsonl

    data = {split: read_jsonl(path / f"{split}.jsonl") for split in ("train", "validation", "test")}
    checks = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("id", "repo", "code_hash", "structural_hash", "query_hash"):
            overlap = {r[key] for r in data[left]} & {r[key] for r in data[right]}
            checks[f"{left}/{right}/{key}"] = len(overlap)
            if overlap:
                raise ValueError(f"Dataset leakage: {left}/{right}/{key}: {len(overlap)}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, default=Path("data/csn-python-v1"))
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=3_000)
    args = parser.parse_args()
    prepare(
        args.source,
        args.output,
        {"train": args.train_size, "validation": args.validation_size, "test": args.test_size},
    )


if __name__ == "__main__":
    main()
