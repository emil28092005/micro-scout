# Index-free MiniCPM search

This experimental mode uses **MiniCPM5-1B Q4_K_M** to choose searches and reads of a
repository's current files. It requires no SQLite index, embeddings, or repository
training. The initial file listing is computed for each request. Model weights
remain loaded in a local Ollama process for 30 minutes after a request.

The default Ollama command uses **off-the-shelf weights**. A separate Transformers
backend supports reference inference and locally trained LoRA adapters. The earlier
fine-tuned MiniLM model remains available through the indexed commands. See the
[measured results](../reports/minicpm5-v1/README.md) before choosing a backend.

## Setup

The initial implementation requires Linux/POSIX, Python 3.11–3.13, `ripgrep` on
PATH, and a running local Ollama server. The filesystem reader uses `openat`,
`O_NOFOLLOW`, and directory descriptors; Windows is not supported by this mode.

```bash
uv sync --extra live --extra mcp --extra dev
ollama pull openbmb/minicpm5:q4_K_M
uv run --no-sync python -m micro_scout.prepare_live
```

`prepare_live` downloads only the approximately 10 MB tokenizer from a pinned
OpenBMB revision, verifies its SHA-256, and caches it in
`~/.cache/micro-scout/minicpm5-tokenizer.json`. Later searches need no network
access beyond the loopback connection to Ollama. Without this cache, the harness
uses a conservative UTF-8 byte bound for context size, which can reject larger
observations. The `live` extra installs the lightweight tokenizer library and
does not install PyTorch or a training stack.

The measured laptop already had Ollama 0.20.4 and the official model downloaded.
The local GGUF blob was verified against OpenBMB's published artifact:

- Repository: `openbmb/MiniCPM5-1B-GGUF`
- Revision: `3d55fac80935ae6456986ad2384b5cbcc4d6c948`
- File: `MiniCPM5-1B-Q4_K_M.gguf`, 688,065,920 bytes
- SHA-256: `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`
- Tokenizer revision: `87179e5c1f455ef22e6223592d2d61351b525bfc`

Ollama tags can change. Verify the model artifact when reproducing the baseline.

## CLI

```bash
uv run --no-sync micro-scout live /path/to/repository \
  "Find where credentials are removed before following a redirect" \
  --trace runs/my-search.json
```

Options include `--max-rounds 6`, `--timeout 90`, `--context 8192`,
`--max-chars 6000`, `--model`, and a loopback-only `--endpoint`. Each model
generation is limited to 512 tokens. All output source ranges are one-based and
inclusive. `max_chars` limits returned source characters, not the entire response
or model tokens. A trace is written only when explicitly requested and contains
the query, generated calls, source observations, errors, and runtime counters.

The `status` field distinguishes `completed`, `abstained`, `budget_exhausted`, and
`model_error`. Completed means the references passed verification; it does not
mean the model found the right implementation. Empty results and failures are
reported without silently falling back to the old indexed search.

## Protocol and tools

The harness uses MiniCPM5's native `<function name="..."><param ...>` syntax,
with no-think ChatML framing. It parses the XML itself through Ollama's raw
generation API. It does not depend on Ollama detecting native tool-call support
in the model's installed template. Sampling uses temperature 0 and seed 42;
these settings do not guarantee bitwise determinism across runtimes.

Available model actions:

| Action | Purpose |
| --- | --- |
| `files(glob)` | List at most 100 visible paths, prioritizing `src/` and `lib/`. |
| `grep(pattern, glob)` | Case-insensitive Rust regex, up to 8 matches per file and 30 returned matches. |
| `read(path, start_line, end_line)` | Read at most 120 numbered lines and approximately 8,000 characters. |
| `finish(path, start_line, end_line)` | Select previously read lines for the caller. |
| `not_found()` | Explicitly finish without evidence. |

The model may issue up to three actions in a response. The first implementation
executes filesystem actions sequentially. It performs one additional initial file
listing, counted in `tool_calls`. This counter counts filesystem action attempts;
`finish` and `not_found` do not increment it. Searches respect the default ignore rules;
positive globs are checked against the default visible-file inventory. A trailing
directory slash in a glob means all files under that directory.

Repository contents are untrusted model input. The executor accepts only the
listed read-only operations, uses argument arrays instead of a shell, rejects
hidden/escaping read paths and symlinks, and caps source files at 1 MB. Each
ripgrep subprocess has a three-second deadline and bounded captured output.
Direct reads of explicitly named non-hidden ignored files are possible; ignore
rules govern search discovery, not access control. The root directory is the
access boundary.

Final references must be covered by this search's read observations. The harness
reopens the source and compares its hash before returning it. Changed files,
invented ranges, and excess source output cause an error that the model can try
to correct within its remaining round budget. This verifies provenance, not
semantic relevance. Token counts use the pinned tokenizer; older exchanges can
be removed with a warning to fit the context. Oversized remaining prompts fail
explicitly rather than relying on silent runtime truncation.

## MCP

```bash
uv run --no-sync micro-scout serve-live /path/to/repository
```

This stdio server exposes `scout_live_search(query, max_chars)` and serializes
requests to the shared local model. Configure the host with absolute paths and a
tool timeout above the chosen search timeout. Source queries are not logged by
the MCP adapter. Starting this server does not replace an existing indexed MCP
configuration.

## Reproducing the development evaluation

The suite in `evals/live-search-v1.json` contains 30 hand-authored English tasks,
10 each for Requests, Flask, and Click. Their revisions and target source hashes
are pinned. Labels were fixed before running the evaluation and are never passed
to the model. The JSON-decoding query used to develop the protocol is excluded.

```bash
mkdir -p data/search-eval
git clone --depth 1 --branch v2.32.5 https://github.com/psf/requests.git data/search-eval/requests
git clone --depth 1 --branch 3.1.2 https://github.com/pallets/flask.git data/search-eval/flask
git clone --depth 1 --branch 8.2.1 https://github.com/pallets/click.git data/search-eval/click
uv run --no-sync python -m micro_scout.eval_live --output runs/live-eval-001
uv run --no-sync python -m micro_scout.eval_live --backend keyword --output runs/keyword-eval-001
```

The evaluator verifies clean repository revisions and source hashes, saves its
configuration before inference, warms the model, records all 30 traces, and
reports failures alongside successes. It measures file hits, target hits (at
least three executable-body lines, or the whole body when shorter), line
precision/recall, end-to-end latency, runtime-reported tokens, and GPU memory
sampled once per second. Function body ranges exclude their leading docstrings;
they are approximate relevance labels and are not exhaustive multi-file context.

These are public, mature Python projects. They may have appeared in MiniCPM's
pretraining, and the tasks were authored during development. This is not a
contamination-free benchmark, a multilingual evaluation, or evidence of improved
coding-task success with Astra. It also does not compare against Astra using grep.

The `keyword` evaluation backend is a fixed, non-neural control: up to 12 literal
term searches, followed by at most three 25-line reads. It ranks windows by
distinct query terms, weighted by their observed match counts. It uses the same
bounded filesystem executor and source verification, without an index. This is
a simple heuristic, not a simulation of a large model choosing and refining
searches. The model backend and keyword control have different action counts;
reports show those counts alongside latency and relevance.

## Reference inference and adapter training

The `policy` extra requires an NVIDIA CUDA GPU for this initial implementation.
Do not run Ollama inference and training on the same 4 GB GPU simultaneously.
Downloading the pinned original checkpoint needs approximately 2.16 GB on disk,
in addition to dependencies and the optional GGUF copy.

```bash
uv sync --extra policy --extra train --extra mcp --extra dev
HF_HUB_DISABLE_XET=1 uv run --no-sync python -m micro_scout.prepare_live --weights
ollama stop openbmb/minicpm5:q4_K_M
uv run --no-sync micro-scout live /path/to/repository "Find retry handling" \
  --backend transformers --bf16
```

The Transformers backend uses the pinned original checkpoint, greedy decoding,
and the same prompt and tool protocol. Without `--bf16` it uses NF4 double
quantization; it keeps the model resident for the lifetime of `serve-live`.
The XML delimiters are special tokens in MiniCPM's tokenizer: they must be
preserved when decoding tool calls. Only terminal end-of-turn tokens are removed.

Prepare the audited CodeSearchNet subset using the indexed model's
[data preparation instructions](TRAINING.md), then build executed demonstrations:

```bash
uv run --no-sync python -m micro_scout.live_data --output data/live-policy-windows-v1
uv run --no-sync python -m micro_scout.train_policy \
  --data data/live-policy-windows-v1 --output runs/minicpm5-policy-v2 --epochs 1
```

This is a small **supervised QLoRA experiment**, not RL or training from scratch.
It uses 256 training trajectories and 32 validation trajectories from disjoint
CodeSearchNet repositories. Requests, Flask, and Click are excluded by repository
name. Each demonstration constructs a three-file synthetic repository, with two
functions per file. It preserves source filenames and varies line offsets and
the position of the target relative to a distractor. An oracle uses the known
label to select a query word and a visible target match. It reads a fixed window
(40 lines before and 60 after the match), then selects the target from the actual
read output. Candidates without an observable match are rejected. Some trajectories
include a failed search before the successful one. Tool observations are real
executor outputs.

Only assistant action tokens and the turn-ending token contribute to training
loss; prompts and source observations are masked. The trainer drops overlength
examples instead of truncating actions, records dataset hashes and settings,
and selects the adapter by validation action loss. It computes output logits
only for the supervised suffix to reduce memory. Tests compare that loss and
its gradients against ordinary masked causal loss.

These demonstrations teach protocol and short search sequences. Oracle-selected
files and final ranges, three-file repositories, and documentation-derived queries
are substantial simplifications. Validation action loss is not repository search accuracy. The
pipeline does not collect an online reward, update a serving model, or establish
improved coding-task performance. No teacher or solver API is called.

Evaluate with identical backend and search settings before and after training:

```bash
uv run --no-sync python -m micro_scout.eval_live \
  --backend transformers --output runs/live-nf4-base
uv run --no-sync python -m micro_scout.eval_live \
  --backend transformers --adapter runs/minicpm5-policy-v2/best \
  --output runs/live-nf4-adapter
uv run --no-sync micro-scout serve-live /path/to/repository \
  --backend transformers --adapter /absolute/path/to/runs/minicpm5-policy-v2/best
```

The earlier `function-ranges` recipe requested exact function boundaries before
reading their contents. Its low validation loss did not translate into target
hits; it is retained only for reproducing the first failed adapter experiment:

```bash
uv run --no-sync python -m micro_scout.live_data \
  --recipe function-ranges --output data/live-policy-v4
uv run --no-sync python -m micro_scout.train_policy \
  --data data/live-policy-v4 --output runs/minicpm5-policy-v1 --epochs 1 --max-length 2048
```

The current default is `read-windows`, with a 2,560-token training limit. Both
recipes are oracle-generated demonstrations, not trajectories from an autonomous
expert agent. The first recipe's prepared train/validation hashes were reproduced
exactly after adding the recipe switch.

Keep the development suite out of training and use a new untouched suite before
selecting a model for deployment. Model binaries, full source traces, and prepared
data stay outside Git. Training currently has no optimizer resume; interrupted
runs should use a new output directory.

References: [MiniCPM5](https://huggingface.co/openbmb/MiniCPM5-1B),
[native chat template](https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/chat_template.jinja),
[SWE-grep](https://cognition.com/blog/swe-grep),
[CodeScout](https://arxiv.org/abs/2603.17829).
