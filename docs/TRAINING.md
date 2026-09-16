# Training and evaluation

## What is trained

Version 0.1 fine-tunes `sentence-transformers/all-MiniLM-L6-v2` as a **shared bi-encoder**: the same small transformer maps queries and code into normalized vectors. This is supervised contrastive training, not training a language model from scratch and not reinforcement learning.

For each batch, the query's paired function is the labeled positive. Other functions in the batch are treated as negatives. The loss averages query-to-code and code-to-query cross-entropy. A temperature scales cosine scores. The model learns ranking without generating text.

At inference, the repository's vectors are already in the index. Each request encodes only its query, compares vectors, and combines rankings with BM25. This makes persistent inference practical without Ollama or vLLM.

## Reproduce the local run

From the repository root:

```bash
uv sync --extra train --extra mcp --extra dev --python 3.12
uv run --no-sync python -m micro_scout.download --output data/source
uv run --no-sync python -m micro_scout.data --source data/source --output data/csn-python-v1
uv run --no-sync python -m micro_scout.train \
  --data data/csn-python-v1 --config configs/laptop.json \
  --output runs/minilm-v1 --device cuda
```

The download helper pins both upstream revisions. Preparation publishes its output only after all three splits pass an overlap audit. It refuses to replace an existing dataset; use a new directory for a changed preprocessing experiment.

The laptop configuration uses 256 code tokens, 96 query tokens, a batch of 24, two epochs, AdamW at `2e-5`, gradient clipping, and mixed precision on CUDA. Full weights are trainable. A 100-minute training limit provides a checkpointed stopping point. Hardware-dependent duration must be measured.

Dataset shuffling is seeded. Exact bitwise reproducibility across PyTorch versions, GPUs, and kernels is not promised. `run.json`, the dataset manifest, and the saved configuration describe the actual experiment.

## Checkpoints and interruption

- `best/`: the checkpoint selected by validation MRR.
- `last/`: the latest checkpoint and optimizer, scheduler, scaler, and RNG state.
- `training.jsonl`: losses, validation scores, timing, and peak allocated GPU memory.
- `result.json`: final run metadata and completion state.

Training checks a stop flag between batches after `SIGINT` or `SIGTERM`, then validates and saves. For a normal stopped run:

```bash
uv run --no-sync python -m micro_scout.train \
  --data data/csn-python-v1 --config configs/laptop.json \
  --output runs/minilm-v1 --resume runs/minilm-v1/last --device cuda
```

Resume requires the same training configuration, dataset manifest, and prepared-file hashes. A weights checksum prevents resuming mismatched weights and optimizer state after an incomplete checkpoint write. This command is intended for the project's own local optimizer states. Model weights use safetensors, and remote custom model code is disabled.

## Evaluation protocol

Checkpoint selection uses 512 fixed validation pairs. The complete validation set is evaluated separately. The final test set is reserved until the model and retrieval settings are frozen.

Every query ranks the same full candidate pool from its split. The comparison includes:

1. BM25 with identifier-aware tokenization.
2. The original pretrained MiniLM encoder.
3. The locally fine-tuned encoder.
4. Hybrid retrieval for each encoder, using the same fixed reciprocal-rank fusion.

All methods receive code without its documentation query. Report MRR, MRR@10, recall@1/5/10, and the candidate count. A paired bootstrap resamples whole repositories to estimate uncertainty in MRR differences while retaining within-repository correlation. Repository-macro MRR is also reported so large projects do not hide performance on smaller ones.

```bash
uv run --no-sync python -m micro_scout.evaluate --split validation --device cuda
uv run --no-sync python -m micro_scout.evaluate --split test --device cuda
```

The metrics and per-query ranks are saved under `runs/minilm-v1/evaluation/`. The evaluator never updates weights. The final test must not become a repeated hyperparameter-selection loop.

## Latency

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-sync micro-scout benchmark \
  --index .micro-scout/index.sqlite --model runs/minilm-v1/best \
  --device cpu --threads 4 --iterations 100 \
  --output runs/minilm-v1/latency.json
```

This measures the warm harness, including search, context assembly, and checking returned files. Startup is reported separately. Five repeated development queries are used; these are not representative production traffic. Index construction is also reported separately.

## What these results cannot establish

CodeSearchNet descriptions are weak task labels. A function may have several valid alternatives, while the metric assumes only one positive. The test does not measure multi-file reasoning, patch correctness, context completeness, or Astra's success rate. A follow-up experiment should compare a fixed solver with and without micro-scout on real, held-out repository tasks.

Luna-based data generation and solver-feedback training are deferred. The first run requires no teacher API key or paid model calls.
