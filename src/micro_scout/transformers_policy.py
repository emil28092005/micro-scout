"""Reference inference and LoRA adapter inference using the original MiniCPM weights."""

from __future__ import annotations

import json
import time
from pathlib import Path

from micro_scout.native_protocol import parse_calls, render_prompt

BASE_ID = "openbmb/MiniCPM5-1B"
BASE_REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"


def decode_action(tokenizer, token_ids):
    # MiniCPM marks its XML delimiters as special tokens. Preserve them and remove
    # only a terminal EOS; skip_special_tokens=True would destroy every tool call.
    ids = list(token_ids)
    if ids and ids[-1] in {1, 130073}:
        ids.pop()
    return tokenizer.decode(ids, skip_special_tokens=False)


class TransformersPolicy:
    def __init__(
        self,
        *,
        adapter: Path | None = None,
        quantized: bool = True,
        context: int = 8192,
        max_tokens: int = 512,
    ):
        import torch
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ValueError("The reference policy currently requires a CUDA GPU")
        if not 2048 <= context <= 32768 or not 64 <= max_tokens <= 2048:
            raise ValueError("Invalid context or generation budget")
        self.context, self.max_tokens = context, max_tokens
        self.model = BASE_ID + (f"+{adapter.name}" if adapter else "")
        self.adapter, self.quantized = adapter, quantized
        torch.set_num_threads(4)
        torch.manual_seed(42)
        path = snapshot_download(BASE_ID, revision=BASE_REVISION, local_files_only=True)
        self.tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        kwargs = {}
        if quantized:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        started = time.monotonic()
        self.network = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
            **kwargs,
        )
        if adapter:
            from peft import PeftModel

            self.network = PeftModel.from_pretrained(self.network, adapter, is_trainable=False)
        self.network.eval()
        self.load_seconds = time.monotonic() - started

    def prompt_tokens(self, messages):
        return len(self.tokenizer.encode(render_prompt(messages), add_special_tokens=False).ids)

    def metadata(self):
        return {
            "model": self.model,
            "base_id": BASE_ID,
            "base_revision": BASE_REVISION,
            "backend": "transformers",
            "quantization": "nf4" if self.quantized else "bf16",
            "adapter": str(self.adapter) if self.adapter else None,
            "load_seconds": self.load_seconds,
        }

    def generate(self, messages, *, timeout=None):
        import torch

        ids = self.tokenizer.encode(render_prompt(messages), add_special_tokens=False).ids
        if len(ids) + self.max_tokens + 32 > self.context:
            raise ValueError("Prompt exceeds context budget")
        inputs = torch.tensor([ids], device="cuda")
        started = time.monotonic()
        with torch.inference_mode():
            output = self.network.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=1,
                eos_token_id=[1, 130073],
                max_time=timeout,
                use_cache=True,
            )
        generated = output[0, len(ids) :].tolist()
        text = decode_action(self.tokenizer, generated)
        try:
            action = parse_calls(text)
        except (ValueError, TypeError) as exc:
            action = {"protocol_error": str(exc)}
        return {
            "response": json.dumps(action),
            "assistant_content": text,
            "done_reason": "length" if len(generated) >= self.max_tokens else "stop",
            "prompt_eval_count": len(ids),
            "eval_count": len(generated),
            "total_duration": int((time.monotonic() - started) * 1e9),
        }
