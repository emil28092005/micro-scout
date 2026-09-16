"""Small QLoRA experiment on executed search actions; all observations are loss-masked."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from importlib.metadata import version
from pathlib import Path

from micro_scout.io import atomic_json, read_jsonl
from micro_scout.live_data import encode_step
from micro_scout.transformers_policy import BASE_ID, BASE_REVISION


def action_loss(model, ids, labels):
    """Compute causal loss for a contiguous supervised suffix without prompt logits."""
    import torch

    supervised = labels[0].ne(-100).nonzero().flatten()
    if ids.shape[0] != 1 or not len(supervised) or int(supervised[0]) < 1:
        raise ValueError("Expected one example with a masked prompt and supervised action")
    first = int(supervised[0])
    if labels[0, first:].eq(-100).any():
        raise ValueError("Supervised action must be a contiguous suffix")
    output = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=ids.shape[1] - first + 1,
        use_cache=False,
    )
    logits = output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1])
    return torch.nn.functional.cross_entropy(logits, labels[:, first:].reshape(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/live-policy-windows-v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/minicpm5-policy-v2"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_steps < 0 or args.max_length < 128 or args.learning_rate <= 0:
        parser.error("Invalid training budget")
    if (args.output / "experiment.json").exists():
        parser.error("Output already exists; choose a new directory")

    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        parser.error("Policy training currently requires an NVIDIA CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = random.Random(42)
    base = snapshot_download(BASE_ID, revision=BASE_REVISION, local_files_only=True)
    tokenizer = Tokenizer.from_file(str(Path(base) / "tokenizer.json"))
    data, dropped = {}, {}
    for split in ("train", "validation"):
        rows = read_jsonl(args.data / f"{split}.jsonl")
        encoded = [
            encode_step(tokenizer, r["messages"], r["action"], args.max_length) for r in rows
        ]
        data[split] = [row for row in encoded if row is not None]
        dropped[split] = len(rows) - len(data[split])
        if not data[split]:
            raise ValueError(f"No usable {split} examples")
    atomic_json(
        args.output / "experiment.json",
        {
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "examples": {k: len(v) for k, v in data.items()},
            "overlength_dropped": dropped,
            "data_sha256": {
                s: hashlib.sha256((args.data / f"{s}.jsonl").read_bytes()).hexdigest()
                for s in ("train", "validation")
            },
            "seed": 42,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "lora_rank": 16,
            "lora_alpha": 32,
            "quantization": "nf4_double_quant",
            "loss": "assistant XML action tokens and end-of-turn token only",
            "data_manifest": json.loads((args.data / "manifest.json").read_text()),
            "packages": {
                p: version(p)
                for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
            },
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
        },
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        quantization_config=quant,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    # Save a portable base reference instead of the local Hugging Face cache path.
    model.peft_config["default"].base_model_name_or_path = BASE_ID
    model.peft_config["default"].revision = BASE_REVISION
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=0.01
    )
    accumulation = 8
    planned = args.epochs * math.ceil(len(data["train"]) / accumulation)
    total_steps = min(planned, args.max_steps) if args.max_steps else planned
    step, best, started = 0, float("inf"), time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    def forward(row):
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return action_loss(model, ids, labels)

    def validate():
        model.eval()
        losses = []
        with torch.no_grad():
            for row in data["validation"]:
                losses.append(float(forward(row)))
        model.train()
        average = sum(losses) / len(losses)
        if not math.isfinite(average):
            raise ValueError("Non-finite validation loss")
        return average

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with (args.output / "metrics.jsonl").open("w") as log:
        initial = {"step": 0, "validation_loss": validate(), "seconds": time.monotonic() - started}
        log.write(json.dumps(initial) + "\n")
        log.flush()
        print(json.dumps(initial), flush=True)
        for epoch in range(args.epochs):
            order = list(data["train"])
            rng.shuffle(order)
            for offset in range(0, len(order), accumulation):
                batch = order[offset : offset + accumulation]
                losses = []
                for row in batch:
                    loss = forward(row)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    (loss / len(batch)).backward()
                    losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row = {
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": sum(losses) / len(losses),
                    "seconds": time.monotonic() - started,
                    "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20,
                }
                if step % 25 == 0 or offset + accumulation >= len(order) or step == total_steps:
                    row["validation_loss"] = validate()
                    if row["validation_loss"] < best:
                        best = row["validation_loss"]
                        model.save_pretrained(
                            args.output / "best",
                            safe_serialization=True,
                            save_embedding_layers=False,
                        )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    model.save_pretrained(
        args.output / "last", safe_serialization=True, save_embedding_layers=False
    )
    atomic_json(
        args.output / "result.json",
        {
            "optimizer_steps": step,
            "trainable_parameters": trainable,
            "training_seconds": time.monotonic() - started,
            "best_validation_loss": best,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "selected_adapter": str(args.output / "best"),
            "note": "Validation measures teacher-forced actions on synthetic repositories, "
            "not task success.",
        },
    )


if __name__ == "__main__":
    main()
