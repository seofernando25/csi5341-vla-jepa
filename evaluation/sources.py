"""Pinned model and simulator-asset revisions from evaluation/sources.json.

Example: python -m evaluation.sources baseline qwen smolvlm world_model pretrain simulator_assets
downloads them; `artifact(name)` resolves a downloaded copy without network access.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from evaluation.common import ROOT, read_json

SOURCES = read_json(ROOT / "evaluation/sources.json")
NAMES = [*SOURCES["models"], "simulator_assets"]


def artifact(name, download=False):
    dataset = name == "simulator_assets"
    source = SOURCES["simulator_assets"] if dataset else SOURCES["models"][name]
    return Path(
        snapshot_download(
            source["repo"],
            revision=source["revision"],
            repo_type="dataset" if dataset else "model",
            local_files_only=not download,
            allow_patterns=None if dataset else ["*.json", "*.safetensors", "*.txt"],
            ignore_patterns=["onnx/*", "original/*"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=NAMES)
    for name in parser.parse_args().names:
        path = artifact(name, download=True)
        print(json.dumps({"name": name, "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
