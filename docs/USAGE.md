# CLI and MCP usage

## Tool behavior

`scout_search` accepts a natural-language query, `top_k` from 1 to 50, and `max_chars` from 100 to 100,000. It returns ranked snippets and up to three graph neighbors within the shared source-character budget. Overlapping source lines are not repeated. Scores are not probabilities.

With a model, the default `mode` is `dense`, selected from the complete validation results. `lexical` uses BM25, which is useful for exact identifiers, and `hybrid` combines the two rankings. Without a model the CLI and MCP server default to `lexical`. The CLI exposes the same choice as `--mode`.

Markdown is excluded from search by default; set `include_docs=true` to include it. An optional `language` filter narrows candidates before rank fusion, for example to `python` or `typescript`. CLI equivalents are `--include-docs` and `--language python`.

Each snippet includes an opaque symbol ID, repository-relative path, line range, source text, file SHA-256, and verification/truncation flags. The response also identifies the index snapshot and model fingerprint. Truncation occurs at line boundaries.

`scout_read` takes an ID returned by the current index. It checks the source hash again before returning a larger range. It accepts no arbitrary filesystem path.

`scout_status` describes the loaded index. `scout_feedback` is available only with an explicit trace path; it accepts IDs actually returned by one of the last 1,000 searches in the current process. Feedback is stored locally and never updates serving weights.

## Example MCP host configuration

Use the executable in the environment where the package was installed. Replace all paths:

```json
{
  "mcpServers": {
    "micro-scout": {
      "command": "/absolute/path/to/micro-scout/.venv/bin/micro-scout",
      "args": [
        "serve",
        "--index", "/absolute/path/to/project/.micro-scout/index.sqlite",
        "--model", "/absolute/path/to/micro-scout/runs/minilm-v1/best",
        "--device", "cpu",
        "--trace", "/absolute/path/to/micro-scout/runs/session.jsonl"
      ],
      "env": {"OPENBLAS_NUM_THREADS": "1"}
    }
  }
}
```

The server uses the official [MCP Python SDK v1 maintenance line](https://py.sdk.modelcontextprotocol.io/v1/), pinned below v2. It speaks stdio and does not open a network listener. The weights and index stay resident for the server's lifetime. A compatible host can call these tools; this alone does not establish downstream model quality.

## Files and index updates

Indexing honors Git's ignored-file rules when available and skips hidden directories, common dependency/build directories, symlinks, unsupported extensions, and files above 1 MB. Python symbols use AST boundaries. Other supported text/code formats use 60-line chunks. Incomplete Python falls back to chunks.

Python functions longer than 48 lines also get 32-line search fragments with an 8-line overlap. The full symbol remains available through each fragment's `parent_id`. This gives the retriever access to code beyond the beginning of a long function. Neural input is still capped at 256 tokens per candidate; very long individual lines can exceed that representation budget. BM25 searches the complete candidate text.

An index is one SQLite file, atomically replaced after a successful build. A model fingerprint prevents combining incompatible query and code embeddings. Reindexing with unchanged weights reuses embeddings for unchanged normalized code.

Returned files are checked against their indexed content hashes. Modified or missing files are excluded with warnings; reads of stale IDs fail. New files require reindexing. This is a snapshot workflow, not a filesystem watcher. Restart the server after rebuilding an index so it loads the new snapshot.

## Trace contents

With `--trace`, the local JSONL trace records queries, returned IDs, snapshot, latency, and feedback. Source snippets are not copied into search trace records. Without this option, query traces are not written. The repository's default `runs/` directory is excluded from Git.

The tool never executes indexed code. Source text is untrusted input for the consuming model and should be treated as data, including any instruction-like comments it contains.
