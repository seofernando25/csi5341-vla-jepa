"""Upstream algorithm sources pinned by repository and revision in studies/compression/methods.json.

Sources are cloned on first use into the ignored outputs/compression/vendor/<method> folder.
"""

import shutil
import subprocess
from pathlib import Path

from evaluation.common import ROOT, read_json

VENDOR = ROOT / "outputs/compression/vendor"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def upstream(method: str) -> tuple[Path, dict]:
    """Return the source folder checked out at the pinned revision and its methods.json entry."""
    entry = read_json(ROOT / "studies/compression/methods.json")["upstream"][method]
    folder = VENDOR / method
    if not folder.exists():
        staging = VENDOR / f".{method}.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", entry["repository"], str(staging))
        git("-C", str(staging), "checkout", "--quiet", entry["revision"])
        staging.rename(folder)
    head = git("-C", str(folder), "rev-parse", "HEAD")
    if head != entry["revision"]:
        raise ValueError(f"{folder} is at {head}, expected pinned revision {entry['revision']}")
    return folder, entry
