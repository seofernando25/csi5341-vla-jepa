"""Host paths of the repository and the fixed mount points inside the `bwrap` sandbox.

Stdlib only, so every rsi module (including `rsi.gpu`, which runs inside the sandbox) can
import it without pulling in anything else.
"""

from __future__ import annotations

from pathlib import Path

RSI_DIR = Path(__file__).resolve().parent
REPO_ROOT = RSI_DIR.parent
STUDY_DIR = REPO_ROOT / "studies" / "program-v1"

SANDBOX_WORK = "/tmp/work"
SANDBOX_OUT = "/tmp/out"
SANDBOX_REPO = "/tmp/repo"
SANDBOX_VENV = "/tmp/venv"
SANDBOX_HF_HUB = "/tmp/.cache/huggingface/hub"
SANDBOX_EVIDENCE = "/tmp/evidence"
SANDBOX_PKGS = "/tmp/pkgs"
SANDBOX_PI_AGENT = "/tmp/pi-agent"
SANDBOX_CLAUDE_CREDENTIALS = "/tmp/.claude/.credentials.json"
