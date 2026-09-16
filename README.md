# micro-scout

A small local code retrieval model and a tool for giving a larger coding model useful source context.

Micro-scout indexes a repository, supports neural, lexical, and hybrid search, and returns **verified file paths, line ranges, and bounded source snippets**. Its MCP server keeps the model in memory between requests.

## Version 0.1

- A shared MiniLM encoder for English queries and source code; code vectors are computed during indexing.
- Local contrastive fine-tuning on a filtered, repository-disjoint Python subset of CodeSearchNet.
- BM25 and hybrid retrieval, Python AST symbols, and conservative static graph neighbors.
- Atomic SQLite snapshots, reusable embeddings, and source-hash checks before returning code.
- CLI and MCP tools: `scout_search`, `scout_read`, `scout_status`, and optional `scout_feedback`.
- Reproducible training, checkpoint resume, baseline comparisons, and tests.

This is an experimental retrieval system. It does not generate patches or train online. The training run and measured results are documented separately; the earlier research roadmap is not a claim that all proposed features have been implemented.

## First measured result

The 22.7M-parameter model was fine-tuned locally on 30,000 filtered CodeSearchNet pairs in **5m 55s** on an RTX 3050 Laptop GPU. On 3,000 held-out function candidates, dense search achieved **70.1% recall@1 / 0.7784 MRR**, versus **58.7% / 0.6819** for the original MiniLM and **46.7% / 0.5595** for BM25. Dense search is the default, chosen on validation before the test.

Warm CPU search measured **18.3 ms median / 23.8 ms p95** on this small development repository; complete MCP calls measured 30.6–34.3 ms. These results do not establish downstream task success with Astra. See the [model card and measured limitations](docs/MODEL_CARD.md) and [experiment artifacts](reports/laptop-v1/README.md). Trained weights are available locally; they are not included in Git or hosted for download yet.

## Install

Python 3.11–3.13 is supported; development uses Python 3.12 and `uv`.

```bash
uv sync --extra train --extra mcp --extra dev --python 3.12
```

The training extra installs PyTorch and can download several gigabytes of CUDA dependencies on Linux. For lexical search alone, `uv sync` is sufficient. For neural inference without data preparation, use `uv sync --extra model --extra mcp`.

## Search a repository

Lexical search works without weights:

```bash
uv run --no-sync micro-scout index /path/to/repository --output .micro-scout/lexical.sqlite
uv run --no-sync micro-scout search "read configuration from a file" \
  --index .micro-scout/lexical.sqlite
```

After training, use the selected checkpoint:

```bash
uv run --no-sync micro-scout index /path/to/repository \
  --model runs/minilm-v1/best --output .micro-scout/index.sqlite

uv run --no-sync micro-scout search "validate the source before returning a reference" \
  --index .micro-scout/index.sqlite --model runs/minilm-v1/best \
  --top-k 6 --max-chars 12000
```

The default inference device is CPU. Add `--device cuda` to use the GPU. `max_chars` limits returned **source characters**, not model tokens or the complete JSON response. A one-shot CLI command includes model startup; use the MCP server to amortize that cost.

Run `index` again after code changes. Unchanged code embeddings are reused when the model fingerprint matches. Restart a resident server after replacing its index. Changed files are rejected by reference validation until reindexed.

## MCP integration

```bash
uv run --no-sync micro-scout serve \
  --index .micro-scout/index.sqlite --model runs/minilm-v1/best \
  --trace runs/session.jsonl
```

This starts a **stdio** MCP server. Configure a compatible host to launch the installed `micro-scout` executable with `serve` and **absolute paths** to the index, model, and optional trace. See [integration details](docs/USAGE.md).

Feedback records usefulness judgments for later analysis. It does not modify serving weights. Query text is recorded only when an explicit trace path is supplied.

## Train and evaluate

```bash
uv run --no-sync python -m micro_scout.download
uv run --no-sync python -m micro_scout.data
uv run --no-sync python -m micro_scout.train --config configs/laptop.json --device cuda
uv run --no-sync python -m micro_scout.evaluate --split validation
# Run the final test only after freezing model and retrieval settings.
uv run --no-sync python -m micro_scout.evaluate --split test
```

See [training and evaluation](docs/TRAINING.md) and the [dataset card](docs/DATASET.md). Weights, raw data, indexes, and run logs stay outside Git. No hosted model calls are required.

## Scope and limitations

- The initial training task is English description-to-Python-function retrieval. Multi-file bug localization, Russian queries, and other languages need separate evaluation.
- Non-Python files use line chunks; their parsing and retrieval quality are not equivalent to the Python path.
- Static graph edges include containment and approximate same-module calls. This is not a complete call graph or a Graphify integration.
- Search scores are rankings, not confidence probabilities. The larger model may need additional reads.
- Faster retrieval and retrieval accuracy do not establish better end-to-end task success with Astra. That experiment remains to be run.

## Development

```bash
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync pytest -q
```

Tests use temporary repositories and an offline tiny model fixture. They do not download training data or call model APIs. Neural tests require the model extra; dataset preparation tests require the train extra.

## Research

[Research and development plan](docs/RESEARCH.md): related work, Graphify, architecture alternatives, data, resources, and the longer-term roadmap. Research date: September 16, 2026. The research budget excludes calls to teacher and solver models. Luna-based generation is deferred from this first local experiment.
