"""Archive exact execution sources by their recorded hashes, excluding runtime data."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from evaluation.common import ROOT, file_hash, read_json, write_json


def archive(run, overrides):
    record = read_json(run / "run.json")
    manifest = record["environment"]["source_manifest"]
    required = {
        "pyproject.toml",
        "uv.lock",
        "evaluation/common.py",
        "evaluation/models.py",
        "evaluation/sources.json",
    }
    required.update(p for p in manifest if p.startswith("src/") and p.endswith(".py"))
    if "requested_steps" in record:
        required.update({"evaluation/train.py", "evaluation/training_config.json"})
    else:
        required.update(
            {"evaluation/run.py", "evaluation/libero_setup.py", "evaluation/protocol.json"}
        )
        if record.get("experiment") == "core_benchmark":
            required.add("evaluation/core_benchmark.py")
    contents = {}
    hashes = {}
    for name in sorted(required):
        path = overrides.get(name, ROOT / name)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if manifest.get(name) != digest:
            raise ValueError(
                f"Current source differs from recorded execution: {name} in {run.name}; supply its exact historical bytes with --override-source"
            )
        contents[name], hashes[name] = content, digest
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    directory = ROOT / "studies/evaluation/sources"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.zip"
    if not target.exists():
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for name, content in contents.items():
                info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                handle.writestr(info, content)
    with zipfile.ZipFile(target) as handle:
        for name, digest in hashes.items():
            if hashlib.sha256(handle.read(name)).hexdigest() != digest:
                raise ValueError("Existing source archive fails integrity verification")
    write_json(
        directory / f"{identity}.json", {"files": hashes, "archive_sha256": file_hash(target)}
    )
    return {
        "archive": str(target.relative_to(ROOT)),
        "sha256": file_hash(target),
        "files": len(hashes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--override-source", action="append", default=[], metavar="REPO_PATH=EXACT_SOURCE_FILE"
    )
    args = parser.parse_args()
    overrides = {name: Path(path) for name, path in (v.split("=", 1) for v in args.override_source)}
    index_path = ROOT / "studies/evaluation/sources/index.json"
    index = read_json(index_path) if index_path.exists() else {}
    for run in args.runs:
        result = archive(run, overrides)
        if run.name in index and index[run.name] != result:
            raise ValueError("An existing run's source archive cannot be replaced")
        index[run.name] = result
        write_json(index_path, index)
        print(run.name, result["archive"])


if __name__ == "__main__":
    main()
