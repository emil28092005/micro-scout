import json
import subprocess

import numpy as np
import pytest

from micro_scout.index import Index, build_index
from micro_scout.lexical import BM25, reciprocal_rank_fusion
from micro_scout.metrics import ranks_from_scores, retrieval_metrics
from micro_scout.scout import Scout, StaleReferenceError
from micro_scout.symbols import build_edges, parse_source, source_paths


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "files.py").write_text('''def read_file(path):
    """Read text from a file."""
    return open(path).read()

def parse_file(path):
    return read_file(path).splitlines()

class Writer:
    def save(self, path, text):
        with open(path, "w") as stream:
            stream.write(text)
''')
    (root / "numbers.py").write_text("def add_numbers(a, b):\n    return a + b\n")
    return root


def test_bm25_and_empty_corpus():
    index = BM25(["read file content", "write file content", "sort numbers"])
    assert np.argmax(index.score("sort numbers")) == 2
    assert np.all(index.score("notpresent") == 0)
    assert BM25([]).score("x").shape == (0,)


def test_stable_ranks_and_single_positive_metrics():
    scores = np.array([[1, 1, 0], [3, 2, 1], [0, 1, 2]], dtype=float)
    ranks = ranks_from_scores(scores)
    assert ranks.tolist() == [1, 2, 1]
    assert retrieval_metrics(ranks)["mrr"] == pytest.approx(5 / 6)
    with pytest.raises(ValueError):
        ranks_from_scores(np.array([[float("nan")]]))


def test_fusion_does_not_create_lexical_matches_for_zero_scores():
    result = reciprocal_rank_fusion(np.zeros(3), np.array([0.2, 0.9, 0.1]))
    assert np.argmax(result) == 1
    assert result[1] == pytest.approx(0.5 / 61)


def test_ranges_decorators_nested_symbols_and_graph():
    text = "@decorator\ndef outer():\n    def inner():\n        return 1\n    return inner()\n"
    symbols = parse_source("a.py", text)
    assert [(s.name, s.start_line, s.end_line) for s in symbols] == [
        ("outer", 1, 5),
        ("outer.inner", 3, 4),
    ]
    assert build_edges(symbols) == [
        {"source": symbols[0].id, "target": symbols[1].id, "kind": "contains"}
    ]


def test_invalid_python_uses_line_chunks():
    symbols = parse_source("broken.py", "def incomplete(\n  unfinished", chunk_lines=1)
    assert [s.kind for s in symbols] == ["chunk", "chunk"]
    assert [s.start_line for s in symbols] == [1, 2]


def test_gitignore_and_symlinks_are_respected(repository, tmp_path):
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("ignored.py\n")
    (repository / "ignored.py").write_text("secret = 1")
    (repository / ".hidden.py").write_text("secret = 2")
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 3")
    (repository / "linked.py").symlink_to(outside)
    names = {p.name for p in source_paths(repository)}
    assert names == {"files.py", "numbers.py"}


def test_search_returns_verified_references_and_budget(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    response = scout.search("read file", mode="lexical", max_chars=200, top_k=2)
    assert response["results"]
    assert response["returned_chars"] <= 200
    assert response["returned_chars"] == sum(
        len(r["content"]) for r in response["results"] + response["neighbors"]
    )
    for item in response["results"] + response["neighbors"]:
        lines = (repository / item["path"]).read_text().splitlines()
        assert item["content"] == "\n".join(lines[item["start_line"] - 1 : item["end_line"]])
        assert item["verified"]


def test_changed_and_deleted_files_never_return_stale_content(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    symbol = next(s for s in scout.index.symbols if s.name == "read_file")
    (repository / "files.py").write_text("# now completely different\n")
    with pytest.raises(StaleReferenceError, match="Source changed"):
        scout.read(symbol.id)
    response = scout.search("read file", mode="lexical")
    assert not response["results"]
    assert response["warnings"]
    (repository / "files.py").unlink()
    with pytest.raises(StaleReferenceError, match="Source unavailable"):
        scout.read(symbol.id)


def test_replacing_source_with_symlink_is_rejected(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    symbol = next(s for s in scout.index.symbols if s.name == "add_numbers")
    outside = tmp_path / "outside.py"
    outside.write_text((repository / "numbers.py").read_text())
    (repository / "numbers.py").unlink()
    (repository / "numbers.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        scout.read(symbol.id)


class FakeEncoder:
    """Deterministic embeddings exercise index integrity, not model quality."""

    fingerprint = "test-encoder-v1"
    dimension = 4

    def __init__(self):
        self.encoded = 0

    def encode(self, texts, **kwargs):
        self.encoded += len(texts)
        return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(texts), 1))


def test_dense_index_reuses_vectors_and_rejects_wrong_model(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    encoder = FakeEncoder()
    first = build_index(repository, path, encoder)
    assert encoder.encoded == first["symbols"]
    second = build_index(repository, path, encoder)
    assert encoder.encoded == first["symbols"]
    assert second["reused_embeddings"] == first["symbols"]
    assert Index(path).vectors.shape == (first["symbols"], 4)
    encoder.fingerprint = "different-model"
    with pytest.raises(ValueError, match="fingerprints differ"):
        Scout(Index(path), encoder)


def test_failed_refresh_keeps_previous_complete_index(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    first = build_index(repository, path)
    with pytest.raises(ValueError, match="exceeds"):
        build_index(repository, path, max_symbols=1)
    assert Index(path).metadata["snapshot"] == first["snapshot"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": "x", "top_k": 0},
        {"query": "x", "max_chars": 1},
        {"query": "x", "mode": "bad"},
        {"query": "x", "mode": "dense"},
    ],
)
def test_search_input_validation(repository, tmp_path, kwargs):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    with pytest.raises(ValueError):
        Scout(Index(path)).search(**kwargs)


def test_feedback_is_logged_without_weight_update(repository, tmp_path):
    path, trace = tmp_path / "index.sqlite", tmp_path / "trace.jsonl"
    build_index(repository, path)
    scout = Scout(Index(path), trace_path=trace)
    result = scout.search("read file", mode="lexical")
    answer = scout.feedback(result["request_id"], [result["results"][0]["id"]], "helpful")
    assert answer == {"recorded": True, "weights_updated": False}
    assert [json.loads(line)["event"] for line in trace.read_text().splitlines()] == [
        "search",
        "feedback",
    ]


def test_feedback_rejects_unknown_requests(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path), trace_path=tmp_path / "trace.jsonl")
    with pytest.raises(ValueError, match="Unknown or expired"):
        scout.feedback("a" * 32, [], "helpful")


def test_context_does_not_repeat_overlapping_source_lines(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    response = scout.search("Writer save write path text", mode="lexical")
    seen = set()
    for item in response["results"] + response["neighbors"]:
        locations = {(item["path"], i) for i in range(item["start_line"], item["end_line"] + 1)}
        assert not seen & locations
        seen |= locations


def test_language_and_documentation_filters(repository, tmp_path):
    (repository / "README.md").write_text("uniquedocumentationneedle")
    (repository / "client.ts").write_text("function uniquetypescriptneedle() { return 1; }")
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    assert not scout.search("uniquedocumentationneedle", mode="lexical")["results"]
    docs = scout.search("uniquedocumentationneedle", mode="lexical", include_docs=True)
    assert docs["results"][0]["path"] == "README.md"
    code = scout.search("uniquetypescriptneedle", mode="lexical", language="typescript")
    assert code["results"][0]["path"] == "client.ts"
    assert not scout.search("uniquetypescriptneedle", mode="lexical", language="python")["results"]


def test_repository_cluster_bootstrap_retains_group_correlation():
    from micro_scout.evaluate import paired_mrr_interval

    result = paired_mrr_interval(
        np.array([1, 1, 1, 10]), np.array([2, 2, 2, 2]), ["a", "a", "a", "b"]
    )
    assert result["delta"] == pytest.approx(0.275)
    assert result["ci95"] == pytest.approx([-0.4, 0.5])
    assert result["repository_clusters"] == 2


def test_file_hash_matches_raw_crlf_bytes_and_detects_line_ending_changes(repository, tmp_path):
    import hashlib

    raw = b"def crlf():\r\n    return 42\r\n"
    source = repository / "crlf.py"
    source.write_bytes(raw)
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    symbol = next(s for s in scout.index.symbols if s.name == "crlf")
    assert scout.read(symbol.id)["file_sha256"] == hashlib.sha256(raw).hexdigest()
    source.write_bytes(raw.replace(b"\r\n", b"\n"))
    with pytest.raises(StaleReferenceError):
        scout.read(symbol.id)


def test_source_that_grows_after_indexing_is_bounded(repository, tmp_path):
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    symbol = next(s for s in scout.index.symbols if s.name == "add_numbers")
    (repository / "numbers.py").write_text("x" * 1_000_001)
    with pytest.raises(ValueError, match="file-size limit"):
        scout.read(symbol.id)


def test_long_function_tail_has_a_searchable_fragment(repository, tmp_path):
    lines = ["def long_function():"] + [f"    value_{i} = {i}" for i in range(90)]
    lines.append("    return unique_tail_marker")
    (repository / "long.py").write_text("\n".join(lines) + "\n")
    path = tmp_path / "index.sqlite"
    build_index(repository, path)
    scout = Scout(Index(path))
    result = scout.search("unique_tail_marker", mode="lexical", top_k=1)
    hit = result["results"][0]
    assert hit["kind"] == "fragment"
    assert "unique_tail_marker" in hit["content"]
    assert hit["start_line"] > 48
    parent = scout.read(hit["parent_id"])
    assert parent["name"] == "long_function"
    assert parent["start_line"] == 1
