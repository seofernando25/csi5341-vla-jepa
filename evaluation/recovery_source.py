"""Create an attributable corrected n0008 source copy without changing frozen RSI."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import file_hash, write_json

OLD = '''            do_rescale=False,
            **({"images_kwargs": {"device": self.model.device}}
               if self.config.image_processor_backend == "torchvision" else {}),'''
NEW = '''            # Keep image arguments together: structured kwargs shadow flat ones.
            images_kwargs={"do_rescale": False,
                           **({"device": self.model.device}
                              if self.config.image_processor_backend == "torchvision" else {})},'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite an existing scientific source copy")
    relative = Path("src/lerobot_policy_vla_jepa_smolvlm/smolvlm_interface.py")
    before = (args.architecture_source / relative).read_text()
    if before.count(OLD) != 1:
        raise ValueError("Expected exactly the observed legacy n0008 processor call")
    source_files = sorted((args.architecture_source / "src").rglob("*.py"))
    base_manifest = {str(p.relative_to(args.architecture_source)): file_hash(p) for p in source_files}
    for p in source_files:
        dest = args.output / p.relative_to(args.architecture_source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    (args.output / relative).write_text(before.replace(OLD, NEW))
    amended_manifest = {name: file_hash(args.output / name) for name in base_manifest}
    record = {"created_at": datetime.now(UTC).isoformat(),
              "parent": "frozen n0008 confirmation source",
              "scope": "separate recovery experiment; frozen RSI source/journal unchanged",
              "change": "Move do_rescale=False into images_kwargs alongside device; prevent double image rescaling.",
              "changed_files": [name for name in base_manifest if base_manifest[name] != amended_manifest[name]],
              "preserved": "Data/split, tiling, resizing, action/world modules, native losses, prediction semantics and weights",
              "base_source_manifest": base_manifest, "source_manifest": amended_manifest}
    write_json(args.output / "amendment.json", record)
    print(record["change"], flush=True)


if __name__ == "__main__":
    main()
