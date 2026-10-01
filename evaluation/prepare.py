"""Download only pinned inference artifacts, excluding alternate formats and notebooks."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

SOURCES = Path(__file__).with_name("sources.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        choices=["baseline", "smolvlm", "qwen", "world_model", "pretrain", "simulator_assets"],
    )
    args = parser.parse_args()
    for key in args.models:
        sources = json.loads(SOURCES.read_text())
        source = (
            sources["simulator_assets"] if key == "simulator_assets" else sources["models"][key]
        )
        path = snapshot_download(
            source["repo"],
            revision=source["revision"],
            repo_type="dataset" if key == "simulator_assets" else "model",
            allow_patterns=None
            if key == "simulator_assets"
            else ["*.json", "*.safetensors", "*.txt"],
            ignore_patterns=["onnx/*", "original/*"],
        )
        print(json.dumps({"model": key, "revision": source["revision"], "path": path}), flush=True)


if __name__ == "__main__":
    main()
