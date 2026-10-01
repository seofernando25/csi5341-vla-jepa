"""Seal the completed local study for transfer, or verify an extracted bundle."""

import argparse
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json


def checked_path(root, name):
    path = root / name
    if (
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Bundle entry escapes its root")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Bundle entry must be an ordinary file: {name}")
    return path


def verify(root, manifest):
    for name, expected in manifest["files"].items():
        path = checked_path(root, name)
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Bundle content differs: {name}")
    return len(manifest["files"])


def verify_execution_sources(root, record):
    manifest = record["environment"]["source_manifest"]
    required = {
        "pyproject.toml",
        "uv.lock",
        "evaluation/common.py",
        "evaluation/models.py",
        "evaluation/run.py",
        "evaluation/libero_setup.py",
        "evaluation/protocol.json",
        "evaluation/sources.json",
    }
    required.update(n for n in manifest if n.startswith("src/") and n.endswith(".py"))
    if record.get("experiment") == "core_benchmark":
        required.add("evaluation/core_benchmark.py")
    for name in required:
        if file_hash(checked_path(root, name)) != manifest.get(name):
            raise ValueError(f"Source differs from measured execution: {name}")


def ready():
    from evaluation.report import final_measurements_ready

    paths = [
        ROOT / f"studies/evaluation/{p}"
        for p in ["analysis/summary.json", "selection.json", "adaptation/summary.json"]
    ]
    records = [read_json(p) if p.exists() else {} for p in paths]
    return final_measurements_ready(*records), records


def seal(archive, manifest_path):
    complete, (analysis, selection, _) = ready()
    if not complete:
        raise ValueError("Finish the local protocol and select S500 before sealing the cloud study")
    if archive.exists() or manifest_path.exists():
        raise FileExistsError("Use new archive/manifest names; preserve prior transfers")
    names = {
        "README.md",
        "LICENSE",
        "NOTICE.md",
        "pyproject.toml",
        "uv.lock",
        "scripts/cloud_setup.sh",
    }
    for folder in ["evaluation", "src"]:
        names.update(
            str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*") if p.suffix == ".py"
        )
    names.update(f"evaluation/{name}.json" for name in ["protocol", "sources", "training_config"])
    for name in [
        "initial_states.json",
        "observation_bank.json",
        "selection.json",
        "budget.json",
        "training_split.json",
        "validation_samples.json",
        "hardware/local_rtx3090.json",
        "analysis/summary.json",
        "analysis/T1_summary.csv",
        "adaptation/summary.json",
    ]:
        names.add("studies/evaluation/" + name)
    for evidence in analysis["evidence"]:
        directory = ROOT / evidence["run"]
        verify_execution_sources(ROOT, read_json(directory / "run.json"))
        names.update(
            str(p.relative_to(ROOT))
            for p in directory.iterdir()
            if p.name == "run.json" or p.suffix == ".csv"
        )
    checkpoint = ROOT / selection["checkpoint"]["artifact"]
    if file_hash(checkpoint / "model.safetensors") != selection["checkpoint"]["sha256"]:
        raise ValueError("Selected checkpoint changed")
    # Training state is unnecessary for this inference-only hardware comparison.
    names.update(
        str(p.relative_to(ROOT))
        for p in checkpoint.iterdir()
        if p.is_file() and p.name != "train_config.json"
    )
    names.add("outputs/evaluation/observations.pt")
    if (
        file_hash(ROOT / "outputs/evaluation/observations.pt")
        != read_json(ROOT / "studies/evaluation/observation_bank.json")["sha256"]
    ):
        raise ValueError("Frozen observation bank changed")
    files = {}
    for name in sorted(names):
        path = checked_path(ROOT, name)
        files[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Completed local study; frozen inference-only cross-hardware transfer",
        "selected_checkpoint_sha256": selection["checkpoint"]["sha256"],
        "files": files,
    }
    write_json(manifest_path, manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w") as handle:
        for name in files:
            handle.add(ROOT / name, arcname=name, recursive=False)
        handle.add(manifest_path, arcname="cloud-transfer-manifest.json", recursive=False)
    print(
        json.dumps(
            {
                "files": len(files),
                "archive_sha256": file_hash(archive),
                "manifest_sha256": file_hash(manifest_path),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "seal", "verify"])
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "outputs/cloud/evaluation-transfer.tar"
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "cloud-transfer-manifest.json")
    args = parser.parse_args()
    if args.command == "check":
        complete, _ = ready()
        print(json.dumps({"local_protocol_ready_to_seal": complete}))
    elif args.command == "seal":
        seal(args.archive, args.manifest)
    else:
        print(json.dumps({"verified_files": verify(ROOT, read_json(args.manifest))}))


if __name__ == "__main__":
    main()
