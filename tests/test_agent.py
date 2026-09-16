import json

import pytest

from micro_scout.agent import search_live
from micro_scout.local_policy import OllamaPolicy


class ScriptedPolicy:
    model = "test-policy"
    context = 8192
    max_tokens = 512

    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def prompt_tokens(self, messages):
        return 100

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return {
            "response": json.dumps(next(self.actions)),
            "prompt_eval_count": 20,
            "eval_count": 10,
            "done_reason": "stop",
        }


def test_search_loop_collects_verified_evidence_and_optional_trace(tmp_path):
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a + b\n")
    ref = {"path": "a.py", "start_line": 1, "end_line": 2}
    policy = ScriptedPolicy(
        [
            {"calls": [{"tool": "grep", "pattern": "add"}], "results": []},
            {"calls": [{"tool": "read", **ref}], "results": []},
            {"calls": [], "results": [ref]},
        ]
    )
    trace = tmp_path / "trace.json"
    result = search_live(tmp_path, "add two values", policy, trace=trace)
    assert result["status"] == "completed"
    assert result["tool_calls"] == 3
    assert result["input_tokens"] == 60
    assert result["results"][0]["content"] == "def add(a, b):\n    return a + b"
    assert len(json.loads(trace.read_text())["history"]) == 3


def test_unread_references_are_rejected_and_model_can_recover(tmp_path):
    (tmp_path / "a.py").write_text("answer = 42\n")
    ref = {"path": "a.py", "start_line": 1, "end_line": 1}
    policy = ScriptedPolicy(
        [
            {"calls": [], "results": [ref]},
            {"calls": [{"tool": "read", **ref}], "results": []},
            {"calls": [], "results": [ref]},
        ]
    )
    result = search_live(tmp_path, "find answer", policy)
    assert result["invalid_actions"] == 1
    assert result["status"] == "completed"
    assert "Read the complete range" in policy.messages[1][-1]["content"]


def test_exhaustion_and_abstention_are_distinct(tmp_path):
    policy = ScriptedPolicy([{"calls": [{"tool": "files"}], "results": []}])
    result = search_live(tmp_path, "unknown code", policy, max_rounds=1)
    assert result["status"] == "budget_exhausted"
    assert result["results"] == []
    policy = ScriptedPolicy([{"calls": [], "results": []}])
    assert search_live(tmp_path, "unknown code", policy)["status"] == "abstained"


def test_malformed_actions_and_tool_errors_counted(tmp_path):
    policy = ScriptedPolicy(
        [
            {"something": "else"},
            {"calls": [{"tool": "shell"}], "results": []},
            {"calls": [], "results": []},
        ]
    )
    result = search_live(tmp_path, "look around", policy)
    assert result["invalid_actions"] == 1
    assert result["tool_errors"] == 1
    assert result["status"] == "abstained"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.org",
        "http://user@localhost",
        "http://127.0.0.1/elsewhere",
        "http://127.0.0.1?x=1",
    ],
)
def test_only_local_model_endpoints_are_supported(endpoint):
    with pytest.raises(ValueError):
        OllamaPolicy(endpoint=endpoint)
