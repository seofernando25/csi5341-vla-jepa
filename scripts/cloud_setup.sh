#!/usr/bin/env bash
# Prepare a fresh Ubuntu GPU instance. No rental, training or benchmark launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---check}"
case "$mode" in
  --check) ;;
  --install)
    source /etc/os-release
    if [[ "${ID:-}" != ubuntu || "$(id -u)" != 0 ]]; then
      echo 'Installation requires a fresh Ubuntu instance with root access.' >&2
      exit 2
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config python3-venv \
      ffmpeg libgl1 libegl1 libgles2 libglib2.0-0 libosmesa6 libglfw3 rsync unzip
    python3 -m venv outputs/cloud/bootstrap-tools
    outputs/cloud/bootstrap-tools/bin/pip install 'uv==0.12.0'
    CSI_UV="$PWD/outputs/cloud/bootstrap-tools/bin/uv"
    "$CSI_UV" python install 3.12.11
    "$CSI_UV" sync --frozen --python 3.12.11 --extra dev --extra eval --extra report
    mkdir -p outputs/cloud
    dpkg-query -W > outputs/cloud/system-packages.txt
    ;;
  *) echo 'Usage: bash scripts/cloud_setup.sh [--check|--install]' >&2; exit 2 ;;
esac
test -x .venv/bin/python
.venv/bin/python - <<'PY'
import importlib.metadata
import tomllib
from pathlib import Path
import torch

lock = tomllib.loads(Path('uv.lock').read_text())
versions = {}
for package in lock['package']:
    versions.setdefault(package['name'].replace('_', '-'), set()).add(package['version'])
for name in ['torch', 'transformers', 'lerobot', 'hf-libero', 'bitsandbytes', 'mujoco', 'robosuite', 'numpy', 'matplotlib']:
    actual = importlib.metadata.version(name)
    if actual not in versions.get(name, set()):
        raise RuntimeError(f'{name} differs from uv.lock')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; CPU fallback is not a matched evaluation')
if not torch.cuda.is_bf16_supported(including_emulation=False):
    raise RuntimeError('Native BF16 is required by the frozen comparison')
print('Locked packages and CUDA/BF16 discovery passed. Policy-kernel and EGL preflight remain required.')
PY
