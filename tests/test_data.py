import json

import pytest

from micro_scout.data import audit_splits, prepare
from micro_scout.io import write_jsonl


def test_overlap_audit_rejects_renamed_clones(tmp_path):
    for split in ("train", "validation", "test"):
        write_jsonl(
            tmp_path / f"{split}.jsonl",
            [
                {
                    "id": split,
                    "repo": split,
                    "code_hash": split,
                    "query_hash": split,
                    "structural_hash": "same-shape",
                }
            ],
        )
    with pytest.raises(ValueError, match="structural_hash"):
        audit_splits(tmp_path)


def test_overlap_audit_passes_disjoint_records(tmp_path):
    for split in ("train", "validation", "test"):
        write_jsonl(
            tmp_path / f"{split}.jsonl",
            [{key: split for key in ("id", "repo", "code_hash", "query_hash", "structural_hash")}],
        )
    assert set(audit_splits(tmp_path).values()) == {0}


def test_dataset_failure_does_not_publish_partial_output(tmp_path):
    pytest.importorskip("pyarrow")
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "missing", output, {"train": 10, "validation": 10, "test": 10})
    assert not output.exists()
    assert not list(tmp_path.glob(".prepare-*"))


def test_json_writer_rejects_non_finite_metrics(tmp_path):
    from micro_scout.io import atomic_json

    destination = tmp_path / "metrics.json"
    atomic_json(destination, {"score": 1.0})
    with pytest.raises(ValueError):
        atomic_json(destination, {"score": float("nan")})
    assert json.loads(destination.read_text()) == {"score": 1.0}
