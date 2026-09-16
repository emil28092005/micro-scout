"""A bounded MiniCPM search loop; source is retrieved on demand, without indexing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from micro_scout.live_tools import LiveRepository
from micro_scout.local_policy import SearchPolicy
from micro_scout.native_protocol import SYSTEM_PROMPT


def search_live(
    root: Path,
    query: str,
    policy: SearchPolicy,
    *,
    max_rounds: int = 6,
    max_chars: int = 6000,
    timeout: float = 90,
    trace: Path | None = None,
) -> dict:
    if not isinstance(query, str) or not 1 <= len(query) <= 2000:
        raise ValueError("Query must contain 1–2000 characters")
    if not 1 <= max_rounds <= 12 or not 200 <= max_chars <= 20000 or not 1 <= timeout <= 600:
        raise ValueError("Invalid search budget")
    started = time.monotonic()
    repo = LiveRepository(root)
    inventory = repo.files()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Repository: {repo.root.name}\nTask: {query}\n"
                f"Source character budget: {max_chars}."
                f"\nInitial file listing: {json.dumps(inventory)}"
            ),
        },
    ]
    history, results, warnings = [], [], []
    status = "budget_exhausted"
    input_tokens = output_tokens = tool_errors = invalid_actions = 0
    tool_calls = 1  # The deterministic initial file listing is part of the search cost.
    for step in range(max_rounds):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            warnings.append("Search deadline reached")
            break
        reminder = f"\nRound {step + 1}/{max_rounds}."
        if step == max_rounds - 1:
            reminder += " Finish now with the best ranges you have read, or empty results."
        current = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + reminder}]
        # Keep system/task and newest complete exchanges. Record every eviction.
        while (
            policy.prompt_tokens(current) + policy.max_tokens + 32 > policy.context
            and len(current) > 4
        ):
            del current[2:4]
            warnings.append("Older search observations removed to bound context")
        round_started = time.monotonic()
        try:
            response = policy.generate(current, timeout=remaining)
        except (OSError, ValueError) as exc:
            warnings.append(f"Model request failed: {exc}")
            status = "model_error"
            break
        input_tokens += response.get("prompt_eval_count", 0)
        output_tokens += response.get("eval_count", 0)
        raw = response.get("response", "")
        record = {
            "round": step + 1,
            "response": raw,
            "assistant_content": response.get("assistant_content", raw),
            "model_seconds": time.monotonic() - round_started,
            "usage": {k: v for k, v in response.items() if k.endswith(("_count", "_duration"))},
        }
        history.append(record)
        messages.append({"role": "assistant", "content": response.get("assistant_content", raw)})
        try:
            if response.get("done_reason") == "length":
                raise ValueError("Action exceeded the generation budget")
            action = json.loads(raw)
            if "protocol_error" in action:
                raise ValueError(action["protocol_error"])
            calls, refs = action["calls"], action["results"]
            if not isinstance(calls, list) or not isinstance(refs, list) or len(calls) > 3:
                raise ValueError("Expected calls and results arrays; at most three calls")
            if calls and refs:
                raise ValueError("Choose either tool calls or final results")
            if not calls:
                results = repo.finish(refs, max_chars)
                status = "completed" if results else "abstained"
                break
            observations, seen_calls = [], set()
            for call in calls:
                key = json.dumps(call, sort_keys=True)
                if key in seen_calls:
                    observations.append({"call": call, "output": {"error": "Duplicate call"}})
                    tool_errors += 1
                    continue
                seen_calls.add(key)
                if time.monotonic() - started >= timeout:
                    raise ValueError("Search deadline reached")
                tool_calls += 1
                output = repo.execute(call)
                tool_errors += int("error" in output)
                observations.append({"call": call, "output": output})
            record["observations"] = observations
            messages.append(
                {"role": "user", "content": json.dumps(observations, ensure_ascii=False)}
            )
        except (ValueError, TypeError, KeyError) as exc:
            invalid_actions += 1
            record["error"] = str(exc)
            messages.append({"role": "user", "content": json.dumps({"error": str(exc)})})
    result = {
        "query": query,
        "root": str(repo.root),
        "model": policy.model,
        "status": status,
        "results": results,
        "warnings": sorted(set(warnings)),
        "elapsed_seconds": time.monotonic() - started,
        "rounds": len(history),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "invalid_actions": invalid_actions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "returned_chars": sum(len(r["content"]) for r in results),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "index_required": False,
        "protocol": "minicpm5-no-think-xml-v1",
    }
    if trace:
        from micro_scout.io import atomic_json

        atomic_json(
            trace,
            {
                "result": result,
                "history": history,
                "initial_inventory": inventory,
                "settings": {
                    "max_rounds": max_rounds,
                    "max_chars": max_chars,
                    "timeout": timeout,
                    "context": policy.context,
                    "max_tokens": policy.max_tokens,
                },
            },
        )
    return result
