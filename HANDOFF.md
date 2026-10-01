# Handoff — October 1, 2026

## Pick up on a Mac

```bash
git clone https://github.com/seofernando25/csi5341-vla-jepa.git
cd csi5341-vla-jepa
```

For an existing clean checkout, run `bash scripts/handoff.sh "$HOME/Documents/Projects/csi5341-vla-jepa"`. It updates by fast-forward only and never starts training. Review code, results and PDFs on macOS; the matched campaign requires a Linux NVIDIA GPU host. Do not substitute Mac/CPU timing for GPU measurements.

## Current evidence

- Active backbone: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. Qwen is evaluation-only.
- Overnight GPU study: 23 measured 500-step screens, six 1,500-step promotions. n0008 loss **0.2762** versus matched SmolVLM baseline **0.3197**. This is held-out prediction loss, not LIBERO success.
- n0008 uses a trainable residual adapter before the final decoder layer, with native backbone weights frozen. Exact candidate and baseline source snapshots: `studies/dream-rsi/sources/`; hashes accompany each snapshot.
- Confirmation reached **5,000 optimizer steps**. Its 200-sample held-out means were total loss **0.3091**, action loss **0.1869**, weighted world-model loss **0.1222**. These use a different validation protocol from the search screens.
- The user stopped its development rollout after **57/100 episodes, zero successes**. The partial run is marked interrupted and its completed episodes are retained. It is not a final evaluation.
- Historical B16/Q8/Q4 and original S500 results used their recorded protocols. Old PIL-processing latency must not be treated as current GPU-processing latency.

Figures: `studies/dream-rsi/figures/`. Compact per-training-run curves and validation means: `studies/evaluation/training/*/metrics_summary.json`. Dashboard: `scripts/rsi_dashboard.py`, `web/rsi-dashboard/`.

## Host-only artifacts

Checkpoints, optimizer/RNG states, datasets, model caches and the active `.rsi` journal are ignored and are **not** restored by a Git clone. The latest reusable n0008 checkpoint is the `005000/pretrained_model` directory in confirmation training run `20261001T132545701205Z-RSI-n0008-train`; transfer it together with its sibling native training-state directory when resuming.

Two GPU Xid79 disconnects occurred on the local RTX3090. BIOS was updated from F64a to F66; another disconnect followed. A 320W diagnostic cap was prepared but not applied because administrator authentication was unavailable. Last recorded limit was 420W. Local confirmation training and the 5k rollout were stopped; user systemd services and their configuration are host-local. Check their actual state before starting anything.

The user authorized one Vast.ai RTX5090 preflight and training within **USD14 total** in the originating chat. This fork owns repository publication only. Check that chat and the live Vast instance list before renting: do not create a duplicate or assume provisioning has completed. Credentials remain outside the repository. `scripts/cloud_setup.sh` supplies the pinned Ubuntu dependency setup; successful CUDA/BF16, model-kernel and EGL checks are required before training.

## Next agent prompt

> Read AGENTS.md, fetch origin/main and read the latest AGENT_BOARD.jsonl. Continue the CSI5341 SmolVLM project from HANDOFF.md, preserving fixed data splits, preprocessing, initialization and scoring. Check cloud ownership/status before any paid action. Audit observation/action processing and inspect representative failures before attributing zero LIBERO success solely to insufficient training. Keep development and final evaluation separate. Use matched baseline training and measure actual cloud throughput before predicting cost. Do not recreate checkpoints from Git, expose credentials, restart stopped local jobs, or use archived studies for Dream-RSI discovery.
