"""Reproducible contrastive fine-tuning on a single laptop GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from micro_scout.encoder import Encoder
from micro_scout.io import atomic_json, read_jsonl
from micro_scout.metrics import ranks_from_scores, retrieval_metrics


def train(data: Path, output: Path, config: dict, device: str, resume: Path | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import get_linear_schedule_with_warmup

    if output.exists() and (output / "run.json").exists() and resume is None:
        raise ValueError("Run already exists; use --resume or choose a new output directory")
    if (
        config["batch_size"] < 2
        or config["epochs"] < 1
        or config["temperature"] <= 0
        or config["eval_every"] < 1
        or config["max_minutes"] <= 0
        or not 0 <= config["warmup_ratio"] < 1
        or config["learning_rate"] <= 0
    ):
        raise ValueError("Invalid training configuration")
    output.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism at the data/order level; CUDA kernels may differ across devices.
    torch.backends.cudnn.benchmark = False
    rows = read_jsonl(data / "train.jsonl")
    validation = read_jsonl(data / "validation.jsonl")[:512]
    manifest = json.loads((data / "manifest.json").read_text())
    data_hashes = {}
    for split in ("train", "validation"):
        actual_hash = hashlib.sha256((data / f"{split}.jsonl").read_bytes()).hexdigest()
        expected = manifest.get("splits", {}).get(split, {}).get("prepared_sha256")
        if expected and actual_hash != expected:
            raise ValueError(f"Dataset checksum mismatch for {split}")
        data_hashes[split] = actual_hash
    if len(rows) < 2 or len(validation) < 2:
        raise ValueError("Need at least two train and validation pairs")
    if {r["repo"] for r in rows} & {r["repo"] for r in validation}:
        raise ValueError("Training and validation repositories overlap")
    encoder = Encoder(
        str(resume) if resume else config["base_model"],
        device=device,
        revision=config["revision"],
        max_length=config["max_length"],
        query_length=config["query_length"],
        threads=config["threads"],
    )
    manifest_hash = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    try:
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_commit = None
    run_info = {
        "config": config,
        "dataset_manifest_sha256": manifest_hash,
        "dataset_file_sha256": data_hashes,
        "source_commit": source_commit,
        "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_pairs": len(rows),
        "validation_pairs_for_selection": len(validation),
        "parameters": encoder.parameter_count,
        "dimension": encoder.dimension,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "objective": "symmetric in-batch contrastive cross-entropy",
        "test_set_used_for_selection": False,
    }
    query_tokens = encoder.tokenize([r["query"] for r in rows], query=True, padding=False)
    code_tokens = encoder.tokenize([r["code"] for r in rows], padding=False)
    run_info["code_at_token_limit_fraction"] = sum(
        len(ids) == config["max_length"] for ids in code_tokens["input_ids"]
    ) / len(rows)
    run_info["query_at_token_limit_fraction"] = sum(
        len(ids) == config["query_length"] for ids in query_tokens["input_ids"]
    ) / len(rows)
    optimizer = torch.optim.AdamW(
        encoder.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    batch_size = config["batch_size"]
    if batch_size < 2 or config["temperature"] <= 0 or config["epochs"] < 1:
        raise ValueError("Invalid batch size, temperature, or epoch count")
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * config["warmup_ratio"]), total_steps
    )
    use_cuda = encoder.device.type == "cuda"
    amp = use_cuda and config.get("mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, start_batch, step, best = 0, 0, 0, -1.0
    optimizer_steps, skipped_optimizer_steps = 0, 0
    if resume:
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        if (
            state["config"] != config
            or state["dataset_manifest_sha256"] != manifest_hash
            or state["dataset_file_sha256"] != data_hashes
        ):
            raise ValueError("Resume config or dataset differs from the saved run")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch, step, best = (
            state["epoch"],
            state["batch"],
            state["step"],
            state["best"],
        )
        torch.set_rng_state(state["torch_rng"])
        optimizer_steps = state["optimizer_steps"]
        skipped_optimizer_steps = state["skipped_optimizer_steps"]
        if use_cuda:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    atomic_json(output / "run.json", run_info)
    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    log = (output / "training.jsonl").open("a", encoding="utf-8", buffering=1)

    def record(event: dict) -> None:
        event = {"step": step, "elapsed_seconds": round(time.monotonic() - started, 3), **event}
        line = json.dumps(event, allow_nan=False)
        log.write(line + "\n")
        print(line, flush=True)

    def evaluate() -> float:
        q = encoder.encode([r["query"] for r in validation], query=True)
        c = encoder.encode([r["code"] for r in validation])
        result = retrieval_metrics(ranks_from_scores(q @ c.T))
        record({"event": "validation", "candidates": len(c), **result})
        return result["mrr"]

    def save_last(epoch: int, batch: int) -> None:
        path = output / "last"
        encoder.save(path)
        state = {
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "optimizer_steps": optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "best": best,
            "config": config,
            "dataset_manifest_sha256": manifest_hash,
            "dataset_file_sha256": data_hashes,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else [],
        }
        torch.save(state, path / "training_state.pt.tmp")
        os.replace(path / "training_state.pt.tmp", path / "training_state.pt")

    try:
        if not resume:
            best = evaluate()
            encoder.save(output / "best")
        running_loss, running_steps = 0.0, 0
        epoch, next_batch = start_epoch, start_batch
        for epoch in range(start_epoch, config["epochs"]):
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed + epoch))
            for batch_number, begin in enumerate(range(0, len(rows), batch_size)):
                if epoch == start_epoch and batch_number < start_batch:
                    continue
                indices = order[begin : begin + batch_size].tolist()
                next_batch = batch_number + 1
                if len(indices) < 2:
                    continue
                encoder.model.train()
                q = encoder.tokenizer.pad(
                    [{k: values[i] for k, values in query_tokens.items()} for i in indices],
                    return_tensors="pt",
                )
                c = encoder.tokenizer.pad(
                    [{k: values[i] for k, values in code_tokens.items()} for i in indices],
                    return_tensors="pt",
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=encoder.device.type, dtype=torch.float16, enabled=amp
                ):
                    qv, cv = encoder.forward(q), encoder.forward(c)
                    scores = (qv @ cv.T) / config["temperature"]
                    target = torch.arange(len(indices), device=encoder.device)
                    loss = (F.cross_entropy(scores, target) + F.cross_entropy(scores.T, target)) / 2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_optimizer_steps += 1
                step += 1
                running_loss += loss.item()
                running_steps += 1
                if step % 25 == 0:
                    record(
                        {
                            "event": "train",
                            "epoch": epoch + 1,
                            "loss": running_loss / running_steps,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "peak_vram_mb": (
                                torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0
                            ),
                            "optimizer_steps": optimizer_steps,
                            "skipped_optimizer_steps": skipped_optimizer_steps,
                        }
                    )
                    running_loss, running_steps = 0.0, 0
                if step % config["eval_every"] == 0:
                    score = evaluate()
                    if score > best:
                        best = score
                        encoder.save(output / "best")
                    save_last(epoch, next_batch)
                if time.monotonic() - started > config["max_minutes"] * 60 or stop_requested:
                    stop_requested = True
                    break
            score = evaluate()
            if score > best:
                best = score
                encoder.save(output / "best")
            save_last(epoch, next_batch)
            if stop_requested:
                break
        result = {
            **run_info,
            "batch_steps": step,
            "optimizer_steps": optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "best_validation_mrr": best,
            "elapsed_seconds": time.monotonic() - started,
            "stopped_early": stop_requested,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0,
        }
        atomic_json(output / "result.json", result)
        record({"event": "complete", "best_validation_mrr": best})
        return result
    finally:
        log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/csn-python-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/minilm-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/laptop.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.data, args.output, json.loads(args.config.read_text()), args.device, args.resume)


if __name__ == "__main__":
    main()
