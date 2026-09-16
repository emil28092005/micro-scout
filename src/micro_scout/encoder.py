"""A shared MiniLM encoder for queries and precomputed source-code vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FORMAT_VERSION = 1


class Encoder:
    def __init__(
        self,
        model: str = BASE_MODEL,
        *,
        device: str = "cpu",
        revision: str | None = None,
        max_length: int = 256,
        query_length: int = 96,
        threads: int = 4,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        local = Path(model).is_dir()
        metadata = Path(model) / "scout_config.json"
        self.config = {
            "format_version": FORMAT_VERSION,
            "base_model": model,
            "revision": revision or (BASE_REVISION if model == BASE_MODEL else None),
            "max_length": max_length,
            "query_length": query_length,
            "pooling": "attention-masked-mean-l2",
            "code_normalization": "strip-python-docstrings-comments-v1",
        }
        if metadata.is_file():
            self.config = json.loads(metadata.read_text())
            if self.config.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported model artifact format")
        kwargs = {} if local else {"revision": self.config["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, **kwargs)
        self.model = AutoModel.from_pretrained(
            model,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="sdpa",
            **kwargs,
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available; use --device cpu")
        self.model.to(self.device).eval()
        self.dimension = self.model.config.hidden_size
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        signature = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode())
        # Weights alone do not identify an encoder: tokenization and model
        # configuration can change the vectors without changing the weights.
        tokenizer_config = json.loads(self.tokenizer.backend_tokenizer.to_str())
        for runtime_option in ("padding", "truncation"):
            tokenizer_config.pop(runtime_option, None)
        model_config = self.model.config.to_dict()
        for provenance in ("_name_or_path", "transformers_version"):
            model_config.pop(provenance, None)
        signature.update(json.dumps(tokenizer_config, sort_keys=True).encode())
        signature.update(json.dumps(self.tokenizer.special_tokens_map, sort_keys=True).encode())
        signature.update(json.dumps(model_config, sort_keys=True).encode())
        if local:
            weights = sorted(Path(model).glob("*.safetensors"))
            if not weights:
                raise ValueError("Local model must contain safetensors weights")
            for weight in weights:
                with weight.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        signature.update(block)
        self.fingerprint = signature.hexdigest()

    def tokenize(self, texts: list[str], *, query: bool = False, padding=True):
        length = self.config["query_length" if query else "max_length"]
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=length,
            return_tensors="pt" if padding else None,
        )

    def forward(self, batch):
        import torch.nn.functional as F

        outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()})
        mask = batch["attention_mask"].to(self.device).unsqueeze(-1)
        # Pool and normalize in float32 even under mixed precision.
        hidden = outputs.last_hidden_state.float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled, p=2, dim=1)

    def encode(self, texts: list[str], *, query: bool = False, batch_size: int = 32) -> np.ndarray:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.model.eval()
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = self.tokenize(texts[start : start + batch_size], query=query)
                vectors.append(self.forward(batch).cpu().numpy())
        return np.concatenate(vectors).astype(np.float32, copy=False)

    def save(self, path: Path) -> None:
        from micro_scout.io import atomic_json

        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        atomic_json(path / "scout_config.json", self.config)
