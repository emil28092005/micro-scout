import json
import os
import subprocess

import pytest

from micro_scout.live_tools import LiveRepository


@pytest.fixture
def live_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "ignored.py").write_text("secret needle\n")
    (root / ".hidden.py").write_text("hidden needle\n")
    (root / "src").mkdir()
    (root / "src" / "client.py").write_bytes(b"def send():\r\n    return 'Needle'\r\n")
    return LiveRepository(root)


def test_live_search_reads_current_files_without_an_index(live_repo):
    assert live_repo.files()["files"] == ["src/client.py"]
    match = live_repo.grep("needle")["matches"]
    assert [(r["path"], r["line"]) for r in match] == [("src/client.py", 2)]
    path = live_repo.root / "src" / "new.py"
    path.write_text("new_needle = 1\n")
    assert len(live_repo.grep("needle")["matches"]) == 2
    path.unlink()
    assert len(live_repo.grep("needle")["matches"]) == 1
    assert not (live_repo.root / ".micro-scout").exists()


def test_live_ignores_rg_configuration_and_treats_pattern_as_argument(live_repo, monkeypatch):
    config = live_repo.root / "rgconfig"
    config.write_text("--hidden\n--no-ignore\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    assert len(live_repo.grep("needle", "*.py")["matches"]) == 1
    assert not live_repo.grep("--help")["matches"]
    output = live_repo.execute({"tool": "grep", "pattern": "["})
    assert "error" in output


def test_live_finish_requires_read_evidence_and_fresh_hash(live_repo):
    ref = {"path": "src/client.py", "start_line": 1, "end_line": 2}
    with pytest.raises(ValueError, match="Read the complete range"):
        live_repo.finish([ref], 1000)
    read = live_repo.read(ref["path"], 1, 2)
    result = live_repo.finish([ref], 1000)[0]
    assert result["content"] == "def send():\n    return 'Needle'"
    assert result["sha256"] == read["sha256"]
    assert result["verified"] is True
    (live_repo.root / ref["path"]).write_text("def replacement():\n    return 2\n")
    with pytest.raises(ValueError, match="Source changed"):
        live_repo.finish([ref], 1000)


@pytest.mark.parametrize("path", ["../outside.py", "/etc/passwd", ".hidden.py", "src/../../a"])
def test_live_rejects_path_escapes(live_repo, path):
    assert "error" in live_repo.execute(
        {"tool": "read", "path": path, "start_line": 1, "end_line": 2}
    )


def test_live_rejects_symlink_parents_and_special_files(live_repo, tmp_path):
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "source.py").write_text("external secret\n")
    (live_repo.root / "linked").symlink_to(outside, target_is_directory=True)
    (live_repo.root / "link.py").symlink_to(outside / "source.py")
    os.mkfifo(live_repo.root / "pipe.py")
    for path in ["linked/source.py", "link.py", "pipe.py"]:
        assert "error" in live_repo.execute(
            {"tool": "read", "path": path, "start_line": 1, "end_line": 2}
        )


def test_live_bounds_reads_and_output(live_repo):
    path = live_repo.root / "src" / "many.py"
    path.write_text("needle = 1\n" * 300)
    assert len(live_repo.grep("needle", "**/many.py")["matches"]) == 8
    with pytest.raises(ValueError, match="120"):
        live_repo.read("src/many.py", 1, 121)
    live_repo.read("src/many.py", 1, 10)
    with pytest.raises(ValueError, match="max_chars"):
        live_repo.finish([{"path": "src/many.py", "start_line": 1, "end_line": 10}], 10)
    assert live_repo.finish([], 1000) == []
    assert "error" in live_repo.execute({"tool": "shell", "command": "touch unwanted"})


def test_live_timeout_and_output_cap_are_explicit(live_repo):
    script = live_repo.root / "fake-rg"
    script.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
    script.chmod(0o755)
    live_repo.rg, live_repo.timeout = str(script), 0.05
    assert live_repo.files()["truncated"] is True
    script.write_text("#!/usr/bin/env python3\nimport sys\nsys.stdout.write('x'*1000000)\n")
    live_repo.timeout = 2
    data, limited = live_repo._run([])
    assert limited
    assert len(data) <= 280000


def test_live_malformed_tool_input_is_reported(live_repo):
    for call in [{}, {"tool": "read"}, {"tool": "grep", "pattern": "a\nb"}]:
        assert "error" in live_repo.execute(call)
    json.dumps(live_repo.files())
