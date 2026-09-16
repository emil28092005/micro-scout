"""MiniCPM5 no-think inference through a local Ollama runtime."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from micro_scout.native_protocol import parse_calls, render_prompt


class SearchPolicy(Protocol):
    model: str
    context: int
    max_tokens: int

    def prompt_tokens(self, messages: list[dict]) -> int: ...

    def generate(self, messages: list[dict], *, timeout: float | None = None) -> dict: ...


class OllamaPolicy:
    def __init__(
        self,
        model: str = "openbmb/minicpm5:q4_K_M",
        *,
        endpoint: str = "http://127.0.0.1:11434",
        context: int = 8192,
        max_tokens: int = 512,
        timeout: float = 60,
        tokenizer: Path | None = None,
    ):
        url = urllib.parse.urlparse(endpoint)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Only a local HTTP Ollama endpoint is supported")
        if url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
            raise ValueError("Endpoint must be a local origin")
        if not 2048 <= context <= 32768 or not 64 <= max_tokens <= 2048:
            raise ValueError("Invalid context or generation budget")
        self.model, self.endpoint = model, endpoint.rstrip("/")
        self.context, self.max_tokens, self.timeout = context, max_tokens, timeout

        self.tokenizer = None
        default_tokenizer = Path.home() / ".cache/micro-scout/minicpm5-tokenizer.json"
        tokenizer = tokenizer or (default_tokenizer if default_tokenizer.exists() else None)
        if tokenizer:
            from tokenizers import Tokenizer

            expected = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
            if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != expected:
                raise ValueError("Tokenizer does not match the pinned MiniCPM5 revision")
            self.tokenizer = Tokenizer.from_file(str(tokenizer))

        # Ignore proxy environment variables for local model requests and reject redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.endpoint + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ValueError("Model response exceeds 2 MB")
            result = json.loads(data)
            if "error" in result:
                raise ValueError(str(result["error"]))
            return result
        except urllib.error.HTTPError as exc:
            raise ValueError(
                f"Ollama HTTP {exc.code}: {exc.read(1000).decode(errors='replace')}"
            ) from exc

    def metadata(self) -> dict:
        data = self.request("/api/show", {"model": self.model})
        return {"model": self.model, "details": data["details"], "model_info": data["model_info"]}

    def prompt_tokens(self, messages: list[dict]) -> int:
        prompt = render_prompt(messages)
        if self.tokenizer:
            return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)
        # Conservative upper bound for this byte-level BPE tokenizer.
        return len(prompt.encode("utf-8"))

    def generate(self, messages: list[dict], *, timeout: float | None = None) -> dict:
        if self.prompt_tokens(messages) + self.max_tokens + 32 > self.context:
            raise ValueError("Prompt exceeds context budget; narrow searches or increase --context")
        old_timeout = self.timeout
        if timeout is not None:
            self.timeout = min(self.timeout, timeout)
        try:
            result = self.request(
                "/api/generate",
                {
                    "model": self.model,
                    "prompt": render_prompt(messages),
                    "raw": True,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "num_ctx": self.context,
                        "num_predict": self.max_tokens,
                        "temperature": 0,
                        "seed": 42,
                        "num_thread": 4,
                        "stop": ["<|im_end|>", "<|endoftext|>"],
                    },
                },
            )
        finally:
            self.timeout = old_timeout
        result["assistant_content"] = result.get("response", "")
        try:
            result["response"] = json.dumps(parse_calls(result["assistant_content"]))
        except (ValueError, TypeError) as exc:
            result["response"] = json.dumps({"protocol_error": str(exc)})
        return result
