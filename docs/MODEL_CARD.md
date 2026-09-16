# Micro-scout v0.1 model card

**Status:** locally trained experimental code retriever, September 16, 2026. The first run completed on a laptop. The code, configuration, dataset audit, and aggregate results are in this repository. The trained checkpoint is currently a local artifact, not a hosted download.

## Intended use

Retrieve Python source functions from short English descriptions and provide verified source locations to a larger coding model through CLI or MCP. The model is a shared MiniLM bi-encoder, not a generative language model: it scores indexed candidates rather than inventing file paths. Paths and line ranges come from the source index.

The model has **22,713,216 parameters**, six transformer layers, 384-dimensional normalized vectors, a 96-token query limit, and a 256-token code limit. All weights were fine-tuned from [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Mean pooling uses the attention mask.

## Training data and procedure

The training set contains 30,000 documentation/function pairs from 6,819 Python repositories in CodeSearchNet. Documentation and comments were removed from the code input. Quality filtering, deterministic sampling, repository caps, and code/query/structural overlap checks are described in the [dataset card](DATASET.md).

The model trained for two epochs with symmetric in-batch contrastive cross-entropy, batch size 24, AdamW at `2e-5`, mixed precision, and seed 17. This was one training run; no teacher API calls, synthetic Luna data, reinforcement learning, or online weight updates were used.

The best checkpoint was selected at batch step **1,750**, using a fixed 512-pair validation subset. Full validation used 2,000 candidates and selected **dense search** as the default before the test was evaluated. The optional hybrid mode uses fixed 50/50 reciprocal-rank fusion; its weights were not tuned. [Frozen experiment](../reports/laptop-v1/frozen-experiment.json)

## Held-out retrieval results

Each of 3,000 test descriptions ranks all 3,000 test functions. The test covers 444 repositories separated from training and validation by the implemented audit. There is one labeled positive per query. These are function-retrieval results, not coding-task completion rates.

| Method | MRR | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.5595 | 46.73% | 66.80% | 72.63% |
| Original MiniLM, dense | 0.6819 | 58.70% | 79.77% | 85.23% |
| Original MiniLM, hybrid | 0.6880 | 59.63% | 79.60% | 84.97% |
| **Micro-scout, dense (default)** | **0.7784** | **70.10%** | **87.47%** | **91.57%** |
| Micro-scout, hybrid | 0.7221 | 64.07% | 81.50% | 86.63% |

MRR is the average reciprocal rank of the paired function; higher is better. Recall@k is the fraction of queries whose paired function appears in the first k results.

Against original dense MiniLM, MRR improved by **0.0965**, with a paired repository-cluster bootstrap 95% interval of **[0.0849, 0.1075]**. Recall@1 improved by **11.40 percentage points**. Dense MRR versus BM25 improved by 0.2189, interval [0.2005, 0.2373]. These intervals describe sampling variation across repository clusters; they do not measure variation across training seeds.

Full-validation MRR was 0.6082 for BM25, 0.7306 for original dense MiniLM, 0.8027 for trained dense search, and 0.7489 for trained hybrid search. This result determined the default before the final test. No model or fusion settings were subsequently tuned on test results.

Machine-readable [test results](../reports/laptop-v1/test-metrics.json) and [validation results](../reports/laptop-v1/validation-metrics.json) include all metrics, fingerprints, and uncertainty estimates.

## Measured resources and latency

Hardware: Intel Core i7-12650H, 32 GB nominal system RAM, NVIDIA RTX 3050 Laptop GPU with 4 GiB VRAM. Software: Python 3.12.13, PyTorch 2.7.1+cu126, Transformers 4.57.6. [Environment](../reports/laptop-v1/environment.json)

| Measurement | Observed value | Scope |
|---|---:|---|
| Training loop | 354.8 s | Includes selection validation and checkpoint saves; excludes setup/tokenization/downloads |
| Optimizer updates | 2,499 | 2,500 batches; AMP skipped one overflow update |
| Peak allocated Torch GPU memory | 1,167 MiB | Allocator metric; excludes driver/runtime overhead |
| FP32 weight file | 90.9 MB | Tokenizer/configuration add about 0.9 MB |
| CPU warm search median / p95 | 18.3 / 23.8 ms | Four Torch threads, one OpenBLAS thread; 100 searches, five repeated queries |
| MCP process start to initialized | 7.33 s | One measurement, includes interpreter and imports; filesystem cache not flushed |
| Complete MCP search calls | 30.6–34.3 ms | Five sequential integration requests over stdio |
| MCP child peak resident memory | 656.6 MiB | Process RSS, including Python/PyTorch and index |
| Initial index construction | 13.3 s | 199 symbols, 27 files, CPU; excludes encoder initialization |

The latency index was this project's source snapshot `e532e1cccfaf6de0f186e39768e035f0f2982121762a5fc8b6cc1c681c6f0f3d`, before the final report was added. It is a small development repository, not a large-repository or concurrent-load benchmark. The model stays in memory for the MCP process lifetime; a one-shot CLI incurs startup again.

The environment occupied approximately 5.4 GiB on disk, prepared/raw data 613 MiB, and training artifacts 349 MiB at measurement time. Shared package and model download caches are additional. Training time alone does not include dependency installation or downloads.

See [latency measurements](../reports/laptop-v1/latency-cpu.json), [MCP smoke results](../reports/laptop-v1/mcp-smoke.json), [training result](../reports/laptop-v1/training-result.json), and [training curve](../reports/laptop-v1/training-curve.jsonl).

## Serving and artifact identity

The local selected checkpoint is `runs/minilm-v1/best/`. Load it with the project's `Encoder` or CLI; this directory uses Hugging Face transformer files plus `scout_config.json`, not a standalone Sentence Transformers pipeline configuration. The model fingerprint covers weights, tokenization, and encoder configuration. The weight SHA-256 is:

```text
3141c072bf1f4d0ba9bfcf10f1c1744f3174993989c5a2728ec3e63bb86a4450
```

The [frozen manifest](../reports/laptop-v1/frozen-experiment.json) records all checkpoint file hashes. Training used source commit `abffddaa5e54c78a5f86308c4a31bb5cc225c077`; final evaluation and fingerprinting used `424aba1d5f50d1aa316009333a2c5cb09b6ca965`. Subsequent report-only updates do not change those experiments.

Follow [training instructions](TRAINING.md) to reproduce the checkpoint and [usage instructions](USAGE.md) to index a repository or launch MCP. No API key is needed. Source references are checked against the current file bytes, and stale files are excluded. Long Python functions also have overlapping searchable fragments.

## Limits and next experiment

- English description-to-Python-function retrieval is the evaluated task. Russian queries, other languages, bug localization, and multi-file context completeness remain unmeasured.
- CodeSearchNet is an older benchmark. Documentation is a proxy for user intent, and several snippets may be valid while only one is labeled positive.
- Base-model pretraining contamination and all semantic near-duplicates cannot be ruled out. The overlap audit checks implemented fingerprints, not complete repository ancestry.
- 19.3% of training code inputs reached the 256-token cap. The model can miss later context. Serving fragments reduce this limitation but were not part of the function-level benchmark.
- Static graph relationships are containment and approximate same-file calls. There is no Graphify integration, dynamic MoE, diffusion architecture, or learned graph traversal in this version.
- The five author-written MCP smoke queries found the expected file within the top six in all five cases, but only three had that file first. This verifies integration and gives a small qualitative example; it is not an independent quality estimate.
- No downstream Astra task-success experiment has been run. MCP compatibility establishes that a host can call the scout, not that the solver will fix more tasks or use fewer tokens.
- Feedback is logged for later analysis. Updating weights while serving would require a separate evaluated training and promotion process.

The next useful experiment is a fixed, held-out repository-task comparison: the same solver with ordinary search versus the same solver with micro-scout, measuring passing tests, missing context, latency, and total solver tokens. Better function retrieval alone is insufficient to claim a downstream gain.

## Licensing and distribution

The base model is published under Apache 2.0. CodeSearchNet includes code from repositories with their own licenses; the project-code license does not replace those source licenses. Dataset provenance is retained locally. This repository does not redistribute the raw corpus or trained binary weights, and this card does not assign a new license to upstream content.
