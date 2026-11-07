"""Upstream algorithm sources pinned by repository and revision in studies/compression/methods.json.

Sources are cloned on first use into the ignored outputs/compression/vendor/<method> folder.
"""

import subprocess
from pathlib import Path

from evaluation.common import ROOT, read_json

VENDOR = ROOT / "outputs/compression/vendor"


def upstream(method: str) -> tuple[Path, dict]:
    """Return the checked-out source folder and its methods.json entry."""
    entry = read_json(ROOT / "studies/compression/methods.json")["upstream"][method]
    folder = VENDOR / method
    if not folder.exists():
        folder.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", entry["repository"], str(folder)], check=True)
        subprocess.run(
            ["git", "-C", str(folder), "checkout", "--quiet", entry["revision"]], check=True
        )
    return folder, entry
