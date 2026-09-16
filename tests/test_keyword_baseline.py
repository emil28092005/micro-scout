from micro_scout.keyword_baseline import search_keywords


def test_keyword_control_returns_verified_ranges_within_file_boundaries(tmp_path):
    (tmp_path / "retry.py").write_text("def retry():\n    return exponential_backoff()\n")
    result = search_keywords(tmp_path, "Find exponential backoff")
    assert result["model"] is None
    assert result["tool_calls"] == 3
    assert len(result["results"]) == 1
    ref = result["results"][0]
    assert ref["start_line"] == 1 and ref["end_line"] == 2 and ref["verified"]
    assert result["input_tokens"] == 0


def test_keyword_control_abstains_when_no_terms_match(tmp_path):
    (tmp_path / "a.py").write_text("print(42)\n")
    assert search_keywords(tmp_path, "unknown implementation")["status"] == "abstained"
