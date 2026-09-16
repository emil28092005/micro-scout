import json

import pytest

from micro_scout.live_data import build_split, candidates, encode_step
from micro_scout.native_protocol import parse_calls
from micro_scout.transformers_policy import decode_action


def test_decode_preserves_special_xml_delimiters():
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["<function", "</function>", "<param", "</param>"])
    ids = tokenizer.encode("<function<param</param></function>", add_special_tokens=False).ids
    assert decode_action(tokenizer, ids + [130073]) == "<function <param </param> </function>"


def test_action_encoding_masks_all_observations_and_never_truncates():
    class CharacterTokenizer:
        def encode(self, text, *, add_special_tokens):
            return type("Encoded", (), {"ids": list(text.encode())})()

    messages = [
        {"role": "system", "content": "Find code"},
        {"role": "user", "content": "Untrusted source with a secret value"},
    ]
    action = '<function name="not_found"></function>'
    row = encode_step(CharacterTokenizer(), messages, action, 10000)
    first = next(i for i, value in enumerate(row["labels"]) if value != -100)
    assert row["labels"][:first] == [-100] * first
    assert bytes(row["labels"][first:]).decode() == action + "<|im_end|>"
    assert row["input_ids"][first:] == row["labels"][first:]
    assert encode_step(CharacterTokenizer(), messages, action, len(row["input_ids"]) - 1) is None


def test_suffix_loss_matches_full_causal_loss_and_gradients():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from micro_scout.train_policy import action_loss

    torch.manual_seed(42)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    model = transformers.LlamaForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, 6, 7, 8]])
    expected = model(input_ids=ids, labels=labels, use_cache=False).loss
    expected.backward()
    gradient = model.lm_head.weight.grad.clone()
    model.zero_grad()
    actual = action_loss(model, ids, labels)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.lm_head.weight.grad, gradient)


def test_oracle_data_uses_executed_searches_and_excludes_evaluation_repos():
    rows = [
        {
            "id": str(i),
            "repo": f"owner/project{i}",
            "path": f"src/number{i}.py",
            "query": f"Calculate number{i} using a numeric expression",
            "code": f"def calculate_number{i}(value):\n"
            f"    result = value * {i + 1}\n    return result",
            "url": "https://example.org/source",
            "code_hash": str(i),
            "source_revision": "abc",
            "split": "train",
        }
        for i in range(4)
    ]
    excluded = {**rows[0], "repo": "psf/requests"}
    assert len(candidates([*rows, excluded])) == 4
    examples, provenance = build_split([*rows, excluded], 2, 42)
    assert len(provenance) == 2
    for example in examples:
        action = parse_calls(example["action"])
        if action["results"]:
            prior_read = json.loads(example["messages"][-1]["content"].split("\nRound")[0])[0]
            assert prior_read["call"]["tool"] == "read"
            assert "error" not in prior_read["output"]
            ref = action["results"][0]
            assert ref["path"] == prior_read["call"]["path"]
            assert prior_read["output"]["start_line"] <= ref["start_line"]
            assert ref["end_line"] <= prior_read["output"]["end_line"]
            assert prior_read["call"]["end_line"] > ref["end_line"]
