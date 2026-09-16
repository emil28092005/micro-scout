"""Download the pinned MiniCPM5 tokenizer for exact, offline context accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
SHA256 = "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81"
WEIGHTS_SHA256 = "7ab8fd86563125929be78aeec8cb3969c7ed2ead3be1ab9d3ec0a9fa69c8660d"


def prepare_weights():
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            "openbmb/MiniCPM5-1B",
            revision=REVISION,
            allow_patterns=["*.json", "*.jinja", "*.safetensors"],
        )
    )
    digest = hashlib.sha256()
    with (directory / "model-00000-of-00001.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Base weights do not match the pinned SHA-256")
    print(json.dumps({"weights": str(directory), "sha256": WEIGHTS_SHA256}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", action="store_true", help="Also download the 2.16 GB base weights"
    )
    args = parser.parse_args()
    if args.weights:
        prepare_weights()
    target = Path.home() / ".cache/micro-scout/minicpm5-tokenizer.json"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print(json.dumps({"tokenizer": str(target), "status": "already_verified"}))
        return
    url = f"https://huggingface.co/openbmb/MiniCPM5-1B/resolve/{REVISION}/tokenizer.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(16_000_001)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Downloaded tokenizer does not match the pinned SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tokenizer-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"tokenizer": str(target), "sha256": SHA256, "status": "downloaded"}))


if __name__ == "__main__":
    main()
