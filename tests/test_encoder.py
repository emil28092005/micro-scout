import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from micro_scout.encoder import Encoder  # noqa: E402
from micro_scout.io import atomic_json, write_jsonl  # noqa: E402
from micro_scout.train import train  # noqa: E402


@pytest.fixture
def tiny_model(tmp_path):
    """Offline random BERT fixture tests plumbing, never used for reported quality."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    root = tmp_path / "tiny-model"
    root.mkdir()
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "read",
        "write",
        "file",
        "sort",
        "numbers",
        "return",
        "open",
        "def",
        "parse",
        "text",
        "a",
        "b",
        "(",
        ")",
        ":",
    ]
    (root / "vocab.txt").write_text("\n".join(words))
    tokenizer = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
    tokenizer.save_pretrained(root)
    torch.manual_seed(17)
    model = BertModel(
        BertConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(root)
    return root


def test_embedding_save_reload_equivalence(tiny_model, tmp_path):
    encoder = Encoder(str(tiny_model), max_length=32, query_length=16)
    texts = ["read file", "sort numbers"]
    vectors = encoder.encode(texts, query=True, batch_size=1)
    assert vectors.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    saved = tmp_path / "saved"
    encoder.save(saved)
    loaded = Encoder(str(saved))
    np.testing.assert_allclose(loaded.encode(texts, query=True), vectors, atol=1e-6)
    assert encoder.fingerprint == loaded.fingerprint


def test_training_updates_weights_and_can_resume(tiny_model, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {"query": "read file", "code": "def read ( ) : return open ( )", "repo": "train"},
        {"query": "sort numbers", "code": "def sort ( a ) : return numbers", "repo": "train"},
        {"query": "write text", "code": "def write ( text ) : return text", "repo": "train"},
        {"query": "parse file", "code": "def parse ( file ) : return file", "repo": "train"},
    ]
    write_jsonl(data / "train.jsonl", rows)
    write_jsonl(data / "validation.jsonl", [{**r, "repo": "validation"} for r in rows[:2]])
    atomic_json(data / "manifest.json", {"fixture": True})
    config = {
        "base_model": str(tiny_model),
        "revision": None,
        "max_length": 32,
        "query_length": 16,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "temperature": 0.05,
        "warmup_ratio": 0,
        "eval_every": 1,
        "seed": 17,
        "threads": 1,
        "max_minutes": 1,
        "mixed_precision": False,
    }
    before = Encoder(str(tiny_model), max_length=32, query_length=16)
    output = tmp_path / "run"
    result = train(data, output, config, "cpu")
    assert result["optimizer_steps"] == 2
    after = Encoder(str(output / "last"))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before.model.parameters(), after.model.parameters(), strict=True)
    )
    resumed = train(data, output, config, "cpu", output / "last")
    assert resumed["optimizer_steps"] == 2
    state = torch.load(output / "last/training_state.pt", weights_only=True)
    assert state["step"] == 2
    assert json.loads((output / "result.json").read_text())["test_set_used_for_selection"] is False
    with torch.no_grad():
        after.model.embeddings.word_embeddings.weight.add_(0.1)
    after.save(output / "last")
    with pytest.raises(ValueError, match="weights and optimizer state"):
        train(data, output, config, "cpu", output / "last")
