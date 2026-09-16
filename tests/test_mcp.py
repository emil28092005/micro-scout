import asyncio
import os
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from micro_scout.index import build_index  # noqa: E402


def test_real_stdio_tool_roundtrip_and_stale_read(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "reader.py"
    source.write_text("def read_file(path):\n    return open(path).read()\n")
    index = tmp_path / "index.sqlite"
    build_index(root, index)

    async def roundtrip():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "micro_scout",
                "serve",
                "--index",
                str(index),
                "--trace",
                str(tmp_path / "trace.jsonl"),
            ],
            env=dict(os.environ),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {
                "scout_search",
                "scout_read",
                "scout_status",
                "scout_feedback",
            }
            result = await session.call_tool("scout_search", {"query": "read file"})
            assert not result.isError
            payload = result.structuredContent
            hit = payload["results"][0]
            assert hit["verified"] and hit["path"] == "reader.py"
            read = await session.call_tool("scout_read", {"symbol_id": hit["id"]})
            assert not read.isError
            feedback = await session.call_tool(
                "scout_feedback",
                {
                    "request_id": payload["request_id"],
                    "useful_ids": [hit["id"]],
                    "outcome": "helpful",
                },
            )
            assert feedback.structuredContent["weights_updated"] is False
            source.write_text("# changed after indexing\n")
            stale = await session.call_tool("scout_read", {"symbol_id": hit["id"]})
            assert stale.isError
            status = await session.call_tool("scout_status")
            assert not status.isError

    asyncio.run(asyncio.wait_for(roundtrip(), timeout=30))


def test_live_stdio_reads_file_changes_without_reindexing(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "answer.py"
    source.write_text("answer = 1\n")
    script = tmp_path / "live_server.py"
    script.write_text("""import json, sys
from pathlib import Path
from micro_scout.live_server import create_live_server

class Policy:
    model = "scripted-offline-policy"
    context = 8192
    max_tokens = 512
    turn = 0
    def prompt_tokens(self, messages):
        return 100
    def generate(self, messages, **kwargs):
        ref = {"path": "answer.py", "start_line": 1, "end_line": 1}
        action = ({"calls": [{"tool": "read", **ref}], "results": []}
                  if self.turn % 2 == 0 else {"calls": [], "results": [ref]})
        self.turn += 1
        return {"response": json.dumps(action), "done_reason": "stop"}

create_live_server(Path(sys.argv[1]), Policy()).run(transport="stdio")
""")

    async def roundtrip():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(script), str(root)],
            env=dict(os.environ),
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            assert [t.name for t in tools.tools] == ["scout_live_search"]
            first = await session.call_tool("scout_live_search", {"query": "find the answer"})
            assert not first.isError
            assert first.structuredContent["results"][0]["content"] == "answer = 1"
            source.write_text("answer = 42\n")
            second = await session.call_tool("scout_live_search", {"query": "find the answer"})
            assert second.structuredContent["results"][0]["content"] == "answer = 42"
            assert not (root / ".micro-scout").exists()

    asyncio.run(asyncio.wait_for(roundtrip(), timeout=30))
