"""Executed, oracle-generated search demonstrations from the audited CodeSearchNet splits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from micro_scout.io import atomic_json, read_jsonl, write_jsonl
from micro_scout.live_tools import LiveRepository
from micro_scout.native_protocol import SYSTEM_PROMPT, render_prompt

EXCLUDED_REPOS = {"requests", "flask", "click"}
STOPWORDS = set(
    [
        "this",
        "that",
        "with",
        "from",
        "into",
        "when",
        "where",
        "which",
        "return",
        "returns",
        "given",
        "there",
        "their",
        "should",
        "would",
        "could",
        "using",
        "used",
        "uses",
        "will",
        "have",
        "make",
        "function",
        "method",
        "object",
        "value",
        "values",
        "parameter",
        "parameters",
        "optional",
        "default",
    ]
)


def candidates(rows: list[dict]) -> list[dict]:
    selected = []
    for row in rows:
        if row["repo"].lower().split("/")[-1] in EXCLUDED_REPOS:
            continue
        try:
            LiveRepository._relative(row["path"])
        except (ValueError, KeyError):
            continue
        code = row["code"]
        if not 3 <= len(code.splitlines()) <= 35 or len(code) > 1800:
            continue
        if not 5 <= len(row["query"].split()) <= 60:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            continue
        terms = sorted(set(re.findall(r"[A-Za-z]{4,}", row["query"].lower())) - STOPWORDS)
        terms = [term for term in terms if re.search(re.escape(term), code, re.IGNORECASE)]
        if not terms:
            continue
        selected.append({**row, "search_terms": terms})
    return selected


def xml_call(name: str, **arguments) -> str:
    params = "".join(
        f'<param name="{key}">{escape(str(value))}</param>' for key, value in arguments.items()
    )
    return f'<function name="{name}">{params}</function>'


def encode_step(tokenizer, messages: list[dict], action: str, max_length: int) -> dict | None:
    prompt = tokenizer.encode(render_prompt(messages), add_special_tokens=False).ids
    completion = tokenizer.encode(action + "<|im_end|>", add_special_tokens=False).ids
    if len(prompt) + len(completion) > max_length:
        return None
    return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}


def build_split(
    rows: list[dict], count: int, seed: int, *, windowed: bool = True
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    pool = candidates(rows)
    rng.shuffle(pool)
    if len(pool) < count + 2:
        raise ValueError("Not enough eligible demonstrations")
    examples, provenance = [], []
    with tempfile.TemporaryDirectory(prefix="micro-scout-live-data-") as directory:
        base = Path(directory)
        for index, target in enumerate(pool):
            if len(provenance) == count:
                break
            root = base / f"example-{index:04d}"
            (root / "src").mkdir(parents=True)
            negatives = rng.sample([r for r in pool if r["id"] != target["id"]], 2)
            snippets = [target, *negatives]
            rng.shuffle(snippets)
            target_path = ""
            written = set()
            for slot, row in enumerate(snippets):
                path = row["path"]
                if path in written:
                    path = f"package_{slot}/{path}"
                written.add(path)
                offset = rng.randint(2, 250)
                prefix = "#\n" * offset
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                if windowed:
                    neighbor = rng.choice([n for n in negatives if n["id"] != row["id"]])
                    neighbor_first = rng.choice([True, False])
                    first, second = (neighbor, row) if neighbor_first else (row, neighbor)
                    content = first["code"].rstrip() + "\n\n" + second["code"].rstrip() + "\n"
                    if neighbor_first:
                        offset += len(neighbor["code"].splitlines()) + 1
                else:
                    content = row["code"].rstrip() + "\n"
                (root / path).write_text(prefix + content)
                if row["id"] == target["id"]:
                    target_path, start, end = (
                        path,
                        offset + 1,
                        offset + len(row["code"].splitlines()),
                    )
            repo = LiveRepository(root)
            inventory = repo.files()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Repository: {target['repo'].split('/')[-1]}\n"
                    f"Task: {target['query']}\nSource character budget: 6000.\n"
                    f"Initial file listing: {json.dumps(inventory)}",
                },
            ]
            # The oracle uses known labels to choose a useful query word. All outputs
            # are obtained by executing the same tools as serving, never invented.
            pattern = max(target["search_terms"], key=len)
            read_start, read_end = start, end
            # A search observation cannot reveal the exact end of unread code.
            # Teach an observable fixed window first, then select a function from
            # the returned source. Every eligible function fits within this window.
            if windowed:
                visible = []
                for pattern in sorted(target["search_terms"], key=lambda t: (-len(t), t)):
                    positive_matches = repo.grep(pattern)["matches"]
                    visible = [
                        m["line"]
                        for m in positive_matches
                        if m["path"] == target_path and start <= m["line"] <= end
                    ]
                    if visible:
                        break
                if not visible:
                    continue  # Never teach a read based on an unobserved match.
                read_start, read_end = max(1, visible[0] - 40), visible[0] + 60
            calls = [
                {"tool": "grep", "pattern": pattern, "glob": ""},
                {
                    "tool": "read",
                    "path": target_path,
                    "start_line": read_start,
                    "end_line": read_end,
                },
            ]
            unknown = [
                w
                for w in re.findall(r"[A-Za-z]{5,}", target["query"])
                if all(w.lower() not in r["code"].lower() for r in snippets)
            ]
            if index % 4 == 0 and unknown:
                calls.insert(0, {"tool": "grep", "pattern": unknown[0], "glob": ""})
            for step, call in enumerate(calls):
                action = xml_call(call["tool"], **{k: v for k, v in call.items() if k != "tool"})
                current = [
                    *messages[:-1],
                    {**messages[-1], "content": messages[-1]["content"] + f"\nRound {step + 1}/6."},
                ]
                examples.append(
                    {
                        "trajectory_id": target["id"],
                        "messages": current,
                        "action": action,
                        "repo": target["repo"],
                    }
                )
                output = repo.execute(call)
                if "error" in output:
                    raise ValueError(f"Demonstration execution failed: {output}")
                messages.extend(
                    [
                        {"role": "assistant", "content": action},
                        {"role": "user", "content": json.dumps([{"call": call, "output": output}])},
                    ]
                )
            ref = {"path": target_path, "start_line": start, "end_line": end}
            repo.finish([ref], 6000)
            messages[-1]["content"] += f"\nRound {len(calls) + 1}/6."
            examples.append(
                {
                    "trajectory_id": target["id"],
                    "messages": messages,
                    "action": xml_call("finish", **ref),
                    "repo": target["repo"],
                }
            )
            provenance.append(
                {
                    "candidate_index": index,
                    **{
                        k: target[k]
                        for k in [
                            "id",
                            "repo",
                            "path",
                            "url",
                            "code_hash",
                            "source_revision",
                            "split",
                        ]
                    },
                    "distractors": [
                        {
                            k: row[k]
                            for k in [
                                "id",
                                "repo",
                                "path",
                                "url",
                                "code_hash",
                                "source_revision",
                                "split",
                            ]
                        }
                        for row in negatives
                    ],
                }
            )
    if len(provenance) != count:
        raise ValueError("Not enough observable demonstrations")
    return examples, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/csn-python-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/live-policy-windows-v1"))
    parser.add_argument("--train-trajectories", type=int, default=256)
    parser.add_argument("--validation-trajectories", type=int, default=32)
    parser.add_argument(
        "--recipe", choices=["read-windows", "function-ranges"], default="read-windows"
    )
    args = parser.parse_args()
    if args.train_trajectories < 1 or args.validation_trajectories < 1:
        parser.error("Trajectory counts must be positive")
    if (args.output / "manifest.json").exists():
        parser.error("Output already exists; use a new directory")
    source_rows = {
        split: read_jsonl(args.source / f"{split}.jsonl") for split in ("train", "validation")
    }
    source_repos = [{r["repo"].lower() for r in candidates(rows)} for rows in source_rows.values()]
    if source_repos[0] & source_repos[1]:
        raise ValueError("Training and validation candidate repositories overlap")
    manifest = {
        "kind": "executed_oracle_demonstrations",
        "seed": 42,
        "limitations": "Three-file synthetic repositories; query-word searches and "
        "oracle-chosen target files and final ranges. Teaches protocol, not realistic planning.",
        "recipe": args.recipe,
        "read_window": {"lines_before_match": 40, "lines_after_match": 60}
        if args.recipe == "read-windows"
        else None,
        "functions_per_file": 2 if args.recipe == "read-windows" else 1,
        "target_position": "randomized before or after a distractor function"
        if args.recipe == "read-windows"
        else "only function",
        "excluded_repositories_by_name": sorted(EXCLUDED_REPOS),
        "splits": {},
    }
    repos = []
    for split, count in [
        ("train", args.train_trajectories),
        ("validation", args.validation_trajectories),
    ]:
        source = args.source / f"{split}.jsonl"
        examples, provenance = build_split(
            source_rows[split], count, 42, windowed=args.recipe == "read-windows"
        )
        write_jsonl(args.output / f"{split}.jsonl", examples)
        write_jsonl(args.output / f"{split}-provenance.jsonl", provenance)
        repos.append({p["repo"] for p in provenance})
        manifest["splits"][split] = {
            "trajectories": len(provenance),
            "action_examples": len(examples),
            "unobservable_candidates_skipped": provenance[-1]["candidate_index"] + 1 - count,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(
                (args.output / f"{split}.jsonl").read_bytes()
            ).hexdigest(),
        }
    if repos[0] & repos[1]:
        raise ValueError("Training and validation repositories overlap")
    manifest["repository_disjoint"] = True
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
