"""Download fixed upstream revisions. No remote dataset or model code is executed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from micro_scout.data import DATASET, REVISION, SOURCE_SHA256
from micro_scout.encoder import BASE_MODEL, BASE_REVISION
from micro_scout.io import atomic_json


def download(output: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for name, info in (
        ("dataset", api.dataset_info(DATASET)),
        ("model", api.model_info(BASE_MODEL)),
    ):
        atomic_json(
            output / f"{name}_metadata.json",
            {
                "id": info.id,
                "current_head_sha": info.sha,
                "downloads": info.downloads,
                "likes": info.likes,
                "author": info.author,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "used_revision": REVISION if name == "dataset" else BASE_REVISION,
            },
        )
    for split in ("train", "validation", "test"):
        destination = output / f"{split}.parquet"
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != SOURCE_SHA256[split]:
                raise ValueError(f"Existing source checksum mismatch: {destination}")
            print(json.dumps({"existing": str(destination)}), flush=True)
            continue
        source = hf_hub_download(
            DATASET,
            f"python/{split}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
        temporary = destination.with_suffix(".parquet.tmp")
        shutil.copyfile(source, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SOURCE_SHA256[split]:
            temporary.unlink()
            raise ValueError(f"Downloaded source checksum mismatch: {split}")
        temporary.replace(destination)
    snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "README.md",
            "LICENSE",
        ],
        max_workers=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
