"""Command-line entry points. Standard output contains results or MCP protocol only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Local source retrieval with verified references")
    sub = parser.add_subparsers(dest="command", required=True)
    index_parser = sub.add_parser("index", help="Build or refresh an atomic repository snapshot")
    index_parser.add_argument("root", type=Path)
    index_parser.add_argument("--output", type=Path, default=Path(".micro-scout/index.sqlite"))
    index_parser.add_argument("--max-symbols", type=int, default=50_000)
    for name in ("search", "serve", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--index", type=Path, default=Path(".micro-scout/index.sqlite"))
        if name == "search":
            command.add_argument("query")
            command.add_argument("--top-k", type=int, default=6)
            command.add_argument("--max-chars", type=int, default=12_000)
            command.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default=None)
            command.add_argument("--no-expand", action="store_true")
            command.add_argument("--language")
            command.add_argument("--include-docs", action="store_true")
        if name in {"search", "serve"}:
            command.add_argument("--trace", type=Path)
        if name == "benchmark":
            command.add_argument("--iterations", type=int, default=100)
            command.add_argument("--output", type=Path)
            command.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default=None)
    for command in (index_parser, *(sub.choices[n] for n in ("search", "serve", "benchmark"))):
        command.add_argument("--model", help="Local weights directory or Hugging Face model ID")
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        from micro_scout.index import Index, build_index
        from micro_scout.scout import Scout

        started = time.perf_counter()
        encoder = None
        if args.model:
            from micro_scout.encoder import Encoder

            encoder = Encoder(args.model, device=args.device, threads=args.threads)
        if args.command == "index":
            result = build_index(args.root, args.output, encoder, max_symbols=args.max_symbols)
        else:
            scout = Scout(Index(args.index), encoder, getattr(args, "trace", None))
            startup_ms = (time.perf_counter() - started) * 1000
            if args.command == "serve":
                from micro_scout.server import create_server

                create_server(scout).run(transport="stdio")
                return
            if args.command == "search":
                result = scout.search(
                    args.query,
                    top_k=args.top_k,
                    max_chars=args.max_chars,
                    mode=args.mode or ("dense" if encoder else "lexical"),
                    expand=not args.no_expand,
                    language=args.language,
                    include_docs=args.include_docs,
                )
                result["startup_ms"] = startup_ms
            else:
                import numpy as np

                if not 1 <= args.iterations <= 10_000:
                    raise ValueError("iterations must be 1–10000")
                queries = [
                    "validate source file hash before returning a code reference",
                    "compute cosine similarity between query and code embeddings",
                    "save model checkpoint after validation improves",
                    "remove documentation strings from Python functions",
                    "combine lexical and neural search rankings",
                ]
                mode = args.mode or ("dense" if encoder else "lexical")
                for query in queries:
                    scout.search(query, mode=mode)
                times = []
                for i in range(args.iterations):
                    t0 = time.perf_counter()
                    scout.search(queries[i % len(queries)], mode=mode)
                    times.append((time.perf_counter() - t0) * 1000)
                result = {
                    "scope": "warm complete harness search including selected-file verification",
                    "iterations": len(times),
                    "query_count": len(queries),
                    "mode": mode,
                    "device": args.device,
                    "threads": args.threads,
                    "symbols": len(scout.index.symbols),
                    "startup_ms": startup_ms,
                    "median_ms": float(np.median(times)),
                    "p95_ms": float(np.quantile(times, 0.95)),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "snapshot": scout.index.metadata["snapshot"],
                    "model_fingerprint": scout.index.metadata.get("encoder_fingerprint"),
                    "note": "Five repeated development queries; not a production workload estimate",
                }
                if args.output:
                    from micro_scout.io import atomic_json

                    atomic_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
