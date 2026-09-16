# Dataset card: CodeSearchNet Python subset v1

## Selection rationale

The original **CodeSearchNet** project was produced by GitHub and Microsoft Research. It supplies function/documentation pairs, source repository identities, and code URLs. The original partitioning separates repositories. This gives a traceable starting point for a code retrieval experiment. [Original project](https://github.com/github/CodeSearchNet)

The local run uses the Parquet conversion hosted at [`code-search-net/code_search_net`](https://huggingface.co/datasets/code-search-net/code_search_net), pinned to revision `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`. The Hugging Face API reported **32,478 downloads and 337 likes** when inspected on September 16, 2026. These are popularity indicators, not accuracy or cleanliness guarantees.

The source dataset is established and attributable, but dates from an older Python ecosystem. Documentation comments are proxy labels; they do not represent the full distribution of coding-agent requests. We prefer this traceable source over an unexplained synthetic collection for the first baseline, then apply our own checks.

The [MiniLM base model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is published by Sentence Transformers under Apache 2.0. The same API inspection reported 254,208,155 downloads and 5,993 likes. It is a general English embedding model; popularity does not establish code-search quality. Its exact revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Preparation

1. Preserve the upstream train/validation/test assignments.
2. Require a repository, source path, and GitHub code URL.
3. Keep queries of 4–80 words and bounded, parseable Python functions.
4. Use the first documentation paragraph as the query.
5. Remove all Python docstrings and comments from the code input while retaining executable string literals.
6. Remove exact token duplicates, normalized structural duplicates, and exact query duplicates from the selected data.
7. Exclude cross-split repository, code, structural, and query matches.
8. Sample with deterministic hash priority and cap each repository at 200 selected examples per split.

Evaluation examples are selected first; overlapping development and training candidates are discarded. The structural hash normalizes identifiers and literals. It is a conservative clone heuristic and may discard distinct functions with similar structure. It cannot guarantee removal of every fork, translated query, or semantic near-duplicate.

The target sample sizes are 30,000 training pairs, 2,000 validation pairs, and 3,000 test pairs. Actual counts, rejection counts, source SHA-256 hashes, prepared-file hashes, and the overlap audit are written to `manifest.json`. A partial preparation run never publishes a completed dataset directory.

The prepared v1 dataset reached those sizes, covering 6,819 training repositories, 397 validation repositories, and 444 test repositories. All 15 pairwise overlap checks passed. The training source contained 412,178 rows; 45,464 failed the quality filters, and 846 additional candidates matched selected evaluation data by structural or query hash. A 12-example training-only spot check found plausible description/function pairs, including networking, file handling, rendering, and configuration. This small review is not a measured label-accuracy estimate.

## Provenance and use

Each prepared example retains its upstream dataset revision, repository, path, URL, content fingerprints, and split. Raw source data is not committed to this repository. CodeSearchNet's project code license does not override the licenses of the underlying repositories; the original project describes per-repository license records. [Source licensing information](https://github.com/github/CodeSearchNet#licenses)

No private local repositories or user conversations are used for this training run. No examples are sent to hosted teacher models.

## Evaluation limits

- The labels identify a paired function, not every valid answer to the query.
- The candidate pool is the selected split, not an entire live repository or a universal code corpus.
- Query language is primarily English. Russian retrieval is not established.
- The base model's pretraining overlap with evaluation examples cannot be excluded.
- Passing the overlap audit establishes the implemented checks only; it does not prove complete independence of repository families.
- Retrieval quality does not establish usefulness to Astra until a paired downstream evaluation is run.
