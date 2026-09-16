# MiniCPM5 index-free search experiment

Recorded September 16, 2026, on an RTX 3050 Laptop GPU with 4 GB VRAM,
an Intel Core i7-12650H, and approximately 30 GiB usable system RAM.

## Conclusion

The index-free search harness, MCP server, and local QLoRA pipeline work, but
this experiment has not produced a useful replacement for a coding model's own
search. Lower validation action loss and source-verified responses did not
translate into reliable implementation localization. The existing indexed
MiniLM integration remains the active MCP configuration.

Both adapters were trained on this laptop without a hosted teacher or solver.
Their weights remain local; this repository publishes the implementation,
training metadata, artifact hashes, and measured development results. No paired
Astra experiment was run, so downstream usefulness with Astra is unproven.

| Variant | Target hits | Correct files | Median / p95 search time |
| --- | ---: | ---: | ---: |
| Original model, Ollama Q4_K_M | 0 / 30 | 2 / 30 | 4.83 / 15.41 s |
| Original model, Transformers NF4 | 0 / 30 | 4 / 30 | 16.39 / 54.43 s |
| QLoRA adapter v1, Transformers NF4 | 0 / 30 | 8 / 30 | 19.40 / 39.98 s |
| QLoRA adapter v2, Transformers NF4 | 0 / 30 | 3 / 30 | 32.88 / 44.77 s |
| Fixed separate-term keyword control | 2 / 30 | 15 / 30 | 0.23 / 0.30 s |

The keyword control is a deliberately simple automatic baseline, not a
measurement of Astra using grep. The model variants are not competitive with
even this control on target retrieval or latency in this development suite.

## Question and setup

Can a local 1B model select `grep` and bounded `read` actions, then return useful
source ranges without building an index? This experiment implements that loop
and a small supervised adapter-training pipeline. It does not measure coding
task success with a larger solver or compare against Astra's own search.

The base model is OpenBMB MiniCPM5-1B, an Apache-2.0 Llama architecture checkpoint
with 1,080,632,832 parameters. The protocol uses native XML function calls and
no-think framing. The harness provides a current file listing, allows six rounds,
at most three actions per round, 512 generated tokens per round, and up to 6,000
returned source characters. It enforces a 90-second search budget; a backend's
current generation or filesystem operation may slightly exceed the deadline.

The frozen development suite has 30 English single-function localization tasks:
10 each from Requests, Flask, and Click. Source commits, file hashes, and target
body ranges are pinned in [`evals/live-search-v1.json`](../../evals/live-search-v1.json).
Targets exclude leading docstrings. A hit needs at least three target body lines
(or the whole body when shorter), and line precision penalizes broad guesses.
Labels are not passed to the search model.

## First frozen baseline: Ollama Q4_K_M

| Measurement | Result |
| --- | ---: |
| Target implementations found | **0 / 30** |
| Correct file returned | 2 / 30 |
| Verified completed responses | 13 / 30 |
| Exhausted search budget | 17 / 30 |
| Median / p95 end-to-end latency | 4.83 / 15.41 s |
| Total invalid actions / tool errors | 37 / 22 |
| Mean rounds / tool calls | 5.37 / 4.70 |
| Sampled peak total GPU memory | 1,213 MiB |

All completed responses had valid source provenance but missed the target body.
This illustrates why successful tool execution is not a retrieval-quality metric.
The unmodified Q4 model is not useful enough to replace the existing indexed
retriever on this suite. A typical failure was selecting a matching comment or
trying to finish without reading the requested range.

Latency excludes model warm-up. GPU memory is device-wide usage sampled once per
second, not an exact allocator peak. This is a small sequential laptop workload,
not a concurrency or production-serving benchmark.

## Reference backend baseline

The corrected Transformers NF4 baseline also found **0/30 targets**, with 4/30
correct files, 13 verified completed responses, and 17 exhausted searches.
Median latency was **16.39 s**, p95 **54.43 s**, and sampled device memory peaked
at **2,198 MiB**. It produced 53 invalid actions and 16 tool errors. These are
the baseline settings used for the adapter comparison; Q4 versus NF4 alone would
confound adapter quality with the inference backend and quantization format.

## Keyword control

A separate non-neural control uses the same executor with up to 12 literal term
searches and three 25-line reads. Windows are ranked by the sum of
`1 / log2(2 + returned matches)` for their distinct query terms. This bounded
match count is a heuristic, not corpus document frequency. The algorithm sees
query text and current files only, never target labels.

The control was added during development after observing the model's failures.
An initial single-OR-query version found 0/30 targets: frequent words exhausted
the result cap before useful matches. Searching terms separately found **2/30**.
Both runs are retained locally. This is a weak automatic keyword baseline; it
does not represent an experienced developer or Astra using adaptive grep.

An idle-laptop repeat of the separate-term control preserved all per-task scores:
**0.229 s median / 0.296 s p95**, with 11.57 tool calls per query on average.
No GPU or model is required. Its macro line precision was only 0.77%, so even
the two target hits do not imply an economical set of source snippets.

## Local adapter training

The selected adapter was trained with QLoRA on the original pinned checkpoint,
using NF4 double quantization and BF16 computation. This is supervised action
training, not a new architecture, full pretraining, or online RL.

| Setting or measurement | Value |
| --- | ---: |
| Train / validation trajectories | 256 / 32 |
| Train / validation action examples | 826 / 104 |
| Epochs / optimizer steps | 1 / 104 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Trainable parameters | 11,206,656 |
| Batch size / gradient accumulation | 1 / 8 |
| Learning rate | 0.0001 |
| Sequence limit / overlength examples dropped | 2,048 / 0 |
| Initial / best validation action loss | 0.33653 / 0.08246 |
| Selected checkpoint step | 100 |
| Training time, including validation and saves | 1,206 s (20m 6s) |
| Peak PyTorch allocated CUDA memory | 2,779 MiB |
| Adapter safetensors size | 44,871,152 bytes |

The timer starts after loading, tokenization, and optimizer setup. Allocator
memory is different from the device-wide sampler used for search runs. The
recorded run used the laptop's existing CUDA stack; no hosted teacher was called.

Data comes from the [audited CodeSearchNet subset](../../docs/DATASET.md), retaining
source and distractor provenance. Complete candidate pools are repository-disjoint
between training and validation; Requests, Flask, and Click are excluded by name.
Three-file synthetic repositories preserve source filenames and vary line offsets.
An oracle executes query-term searches, source reads, and final range selections.
Only assistant action tokens receive loss. Such examples teach protocol and
short search sequences, not realistic repository exploration.

Selection used validation action loss, not the 30-task localization scores.
The selected weights are local at `runs/minicpm5-policy-v1/best`; weights are not
published with the repository. The adapter's base-reference metadata was
normalized to the official model ID and revision after training, without changing
the numerical weights. Future training runs save those portable fields directly.
The exact training source snapshot remains beside the local run.

### Adapter v1 search result and data revision

The first adapter still found **0/30 targets**, despite improving file hits to
8/30 and verified completions to 25/30. Invalid actions fell from 53 to 13; tool
errors fell from 16 to 14. Median latency was 19.40 s and p95 39.98 s with the
unmerged PEFT adapter. Better protocol execution did not produce useful target
body retrieval. These development timings are not isolated kernel benchmarks.

Inspection exposed a weakness in the first oracle recipe: it required exact
unread function boundaries after a short grep observation. That observation
cannot reveal the function's end. The learned policy often copied the same match
line into both read boundaries. The revised recipe instead teaches a fixed
observable window around a match, followed by selection from the actual read.
Two functions per file and randomized target position reduce the shortcut of
always selecting the only function or the end of the file. One unobservable
training candidate was rejected; validation rejected none.

This revision was motivated by development traces, so subsequent localization
results remain development results, not a fresh untouched evaluation.

### Adapter v2 training

The corrected recipe retained 256 training and 32 validation trajectories
(826 / 104 action examples). The limit increased to 2,560 tokens; all examples
fit without truncation or dropping. The same base, LoRA configuration, learning
rate, seed, and one-epoch schedule were used. The run took **1,297 s (21m 37s)**,
with **2,842 MiB** peak PyTorch allocated CUDA memory. Validation action loss
fell from **0.48502 to 0.06406**, selecting step **104**.

These loss values refer to the revised examples and cannot be compared directly
with v1's validation loss. The selected adapter is local at
`runs/minicpm5-policy-v2/best`. See `training-v2-experiment.json`,
`training-v2-result.json`, and `adapter-v2-manifest.json` for the exact settings
and artifact hashes. Both adapters use unmerged PEFT inference in the reference
backend; the reported search latency is not an optimized merged-GGUF deployment.

The v2 evaluation process was interrupted after 28 task results had been saved.
The remaining two tasks were resumed after verifying the frozen code, suite,
repository, and adapter hashes, with a new warm-up. The earlier results were
retained. GPU samples for the first segment were not persisted, so this report
does not claim a full-run GPU peak for v2 inference.

### Adapter v2 search result

The revised adapter again found **0/30 targets** and returned the correct file
on 3/30 tasks. Only 7 responses completed; 23 exhausted the search budget. Median
latency was **32.88 s**, p95 **44.77 s**, with 51 invalid actions and 22 tool errors.
The lower validation action loss did not generalize to this search workload.
Observed failures included poor search terms, matches in documentation instead
of implementation, invalid source ranges, and failure to recover from tool errors.

A diagnostic over all successful intermediate reads found target-body coverage
on 1/30 tasks for Q4, 0/30 for NF4 and adapter v1, and 1/30 for adapter v2. These
are not final-return scores and can consume more context than the returned
snippet budget. They show that the problem starts before final selection: most
searches never read the target implementation. See [`read-coverage.json`](read-coverage.json).

## Artifact integrity and diagnostics

- Base revision: `87179e5c1f455ef22e6223592d2d61351b525bfc`.
- Original safetensors file: 2,161,290,912 bytes; SHA-256
  `7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d`.
- Official Q4_K_M GGUF revision: `3d55fac80935ae6456986ad2384b5cbcc4d6c948`.
- GGUF file: 688,065,920 bytes; SHA-256
  `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`.
- Suite SHA-256: `d06effdf41ee2e38bf8f44629949d9b4491067986fc5a73aef22cf80c5a5bf96`.

Both model-file hashes were verified locally against the publisher's artifacts.
The pinned tokenizer and Ollama agreed on the token count of an actual 611-token
prompt. A separate JSON-decoding development query, excluded from the 30-task
suite, also failed within six rounds with BF16 reference inference (51.18 s).
A one-query thinking-mode diagnostic failed too (18.50 s); it is not a benchmark
of thinking mode. These checks do not establish that every prompt or runtime
configuration would perform equally poorly.

An early Transformers diagnostic mistakenly removed special XML delimiter tokens
during decoding. It was invalidated, corrected, and covered by a regression test.
The published Q4 result used Ollama and was unaffected. Only corrected Transformers
runs should be used for model comparisons.

## MCP integration check

A real stdio client initialized `serve-live` with adapter v2, discovered
`scout_live_search`, and made two calls to the same resident server. Both returned
valid structured responses without MCP transport errors and with
`index_required: false`. Search took 31.89 s and 30.79 s. Both exhausted their
six-round budgets on the excluded JSON-decoding development query and returned
no source ranges. This validates resident inference and the MCP transport, not
successful retrieval. The compact record is [`mcp-smoke.json`](mcp-smoke.json).

## Reproducibility and limits

Implementation validation passed: **73 tests**, Ruff lint and format checks,
lockfile validation, and source/wheel builds. Tests cover bounded filesystem
actions, source verification, protocol decoding, loss masking, and an MCP stdio
round trip. These checks validate implementation behavior, not model search quality.

See [setup and commands](../../docs/LIVE_SEARCH.md). Public JSON artifacts contain
configuration, aggregate results, and per-task scores without full source snippets
or model-generated reasoning. Full traces, source snapshots, weights, and prepared
training data remain in ignored local directories.

This is a development suite on mature public Python repositories, which may appear
in the base model's pretraining. It is not an untouched final holdout. Thirty tasks
do not establish performance across repositories, languages, or coding-agent
workloads. A gain on this suite would need confirmation on new tasks and a paired
solver-with/without-scout experiment before deployment.

## Recommended next experiment

Keep this version experimental. First move range arithmetic into the harness:
let a policy choose an observed match or span handle, and let the executor expand
and validate the corresponding source window. Then train on executed search
trajectories in realistic repositories, including unsuccessful searches,
reformulations, and distractors. The current short oracle demonstrations mainly
teach how to issue actions.

Freeze a new repository-disjoint evaluation before tuning further. Measure target
coverage, returned context size, and latency against lexical and indexed controls.
Only then test a larger solver with and without the scout on the same coding
tasks. More adapter epochs, dynamic experts, or online reward updates are not
supported as the next priority by these results.
