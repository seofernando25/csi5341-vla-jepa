"""What the policy-development agent sees in one step: its prompt, the replay script and the sandbox mounts."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

from rsi.paths import (
    RSI_DIR,
    SANDBOX_PKGS,
    SANDBOX_REPO,
    SANDBOX_VENV,
)
from rsi.sandbox import (
    repo_source_binds,
)

POLICY_DEV_PROMPT = RSI_DIR / "prompts" / "policy_dev.md"
# Repo source the policy agent may read, minus the held-out world generator: the held-out
# worlds are reported next to V and must not be reproducible from inside the workspace.
POLICY_DEV_SOURCE_EXCLUDE = ("rsi/synth.py",)
PKG_DIR_NAME = "pkgs"
REPLAY_SCRIPT_NAME = "replay.py"
UV_BIN = "/tmp/bin/uv"
REPAIR_SUFFIX = (
    "\n\n### Previous attempt validation failure\n{error}\n\n"
    "Fix every violation in `/tmp/work/method.py` cleanly."
)
NOT_BETTER_SUFFIX = (
    "\n\n### Previous attempt did not improve V\n"
    "Its mean V {mean_v:.6f} is not above the incumbent's {best:.6f} "
    "(delta {delta:+.6f}).\n\n"
    "Make one different, larger change in `/tmp/work/method.py`. Run "
    "`python /tmp/work/replay.py` and keep the change only if mean V rises above {best:.6f}."
)
REPLAY_SCRIPT = '''"""Score a candidate policy on this step's recorded trees, exactly as the harness does.

Usage: python replay.py [POLICY]   (default: method.py in this workspace)

Prints V per world and the mean V the harness would record. The harness reruns the replay
itself after you finish; only its result counts.
"""

import os
import sys
from pathlib import Path

from rsi.replay_worker import run_worker

MAX_PARALLELISM = __MAX_PARALLELISM__
LAMBDA_PARALLEL = __LAMBDA_PARALLEL__


def main() -> int:
    evidence = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
    workspace = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
    worlds = sorted((evidence / "worlds").glob("*.json"))
    policy = Path(sys.argv[1]) if len(sys.argv) > 1 else workspace / "method.py"
    try:
        result, _ = run_worker(
            policy, worlds, max_parallelism=MAX_PARALLELISM, lambda_parallel=LAMBDA_PARALLEL
        )
    except Exception as exc:
        print(f"replay failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for world_id, world in result["worlds"].items():
        print(
            f"{world_id}: V={world['v']:.6f} hypervolume={world['hypervolume']:.6f} "
            f"penalty={world['penalty']:.6f}"
        )
    print(f"mean V = {result['mean_v']:.6f} over {len(result['worlds'])} world(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def replay_script(max_parallelism: int, lambda_parallel: float) -> str:
    return REPLAY_SCRIPT.replace("__MAX_PARALLELISM__", str(int(max_parallelism))).replace(
        "__LAMBDA_PARALLEL__", repr(float(lambda_parallel))
    )


def build_prompt(
    *,
    cycle_id: str,
    step: int,
    steps: int,
    max_parallelism: int,
    lambda_parallel: float,
    minutes: float,
    prompt_path: Path,
) -> str:
    template = Path(prompt_path).read_text(encoding="utf-8")
    return template.format(
        step=step,
        steps=steps,
        cycle_id=cycle_id,
        lambda_parallel=lambda_parallel,
        max_parallelism=max_parallelism,
        minutes=f"{minutes:.0f}",
    )


def agent_sandbox(step_dir: Path, study_dir: Path | None = None) -> dict[str, Any]:
    """Extra mounts and env for the policy agent: repo source, GPU, and a writable package dir."""
    pkgs = Path(step_dir) / PKG_DIR_NAME
    pkgs.mkdir(parents=True, exist_ok=True)
    binds: dict[Path, str] = {}
    path = [f"{SANDBOX_VENV}/bin", "/usr/bin"]
    uv = shutil.which("uv")
    if uv:
        # `uv` is a host tool, not a venv entry point; the agent needs it to install packages.
        binds[Path(uv)] = UV_BIN
        path.insert(0, str(Path(UV_BIN).parent))
    return {
        "source_ro": repo_source_binds(study_dir=study_dir, exclude=POLICY_DEV_SOURCE_EXCLUDE),
        "writable_dirs": {pkgs: SANDBOX_PKGS},
        "extra_ro_binds": binds,
        "env_overrides": {
            "PATH": ":".join(path),
            "PYTHONPATH": f"{SANDBOX_PKGS}:{SANDBOX_REPO}",
            "UV_CACHE_DIR": "/tmp/.cache/uv",
            "UV_LINK_MODE": "copy",
        },
    }


def workspace_bytes(root: Path) -> int:
    """Total size of the regular files in one workspace; symlinks are not followed."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total
