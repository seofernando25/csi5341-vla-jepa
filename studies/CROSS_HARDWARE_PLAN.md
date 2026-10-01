# Later cross-architecture evaluation

**Status:** preparation only; run after the local study. No Vast.ai instance has been searched for or rented. Select the offer and freeze the spending/runtime limits when access is provided. Dream-RSI remains deferred.

**Question:** Do the four fixed policies' memory, latency, and manipulation trade-offs persist on a different NVIDIA architecture?

## Machine selection

Local reference: [RTX 3090 / Ampere, sm_86](HARDWARE.md). First candidate class: **RTX 4090 / Ada, sm_89**, retaining 24 GiB VRAM. Alternatives are Hopper H100 (`sm_90`) or Blackwell RTX 5090 (`sm_120`), subject to cost and kernel compatibility. An A40 or another RTX 30-series card would not provide the intended architecture-family change. Verify the actual SKU and compute capability against [NVIDIA's table](https://developer.nvidia.com/cuda/gpus); these are candidate classes, not claims about current Vast availability.

At selection time, compare verified, on-demand offers with one exclusive GPU, at least 24 GiB VRAM, preferably 16 allocated logical CPUs and 64 GiB RAM, adequate shared memory, and a compatible NVIDIA driver. Start with a **150 GB disk estimate**, then size it from the sealed bundle, model cache, environment and output requirements plus headroom. Record effective container allocations, PCIe link, power cap, storage/network characteristics, reliability and full price breakdown. Freeze an hourly cap, total spend cap and maximum runtime before renting. Include setup, storage and transfer charges in actual cost. [Vast offer fields and storage allocation](https://docs.vast.ai/guides/instances/choosing/find-and-rent).

## Freeze and transfer

1. Complete local E1–E3 and select S500 using the existing held-out-loss rule. **No retraining or checkpoint reselection on the rented GPU.**
2. Seal the final code, lockfile, scientific manifests, selected S500 inference checkpoint/processors, and exact 100-observation bank. Transfer actual working-tree source, not just the old Git revision: the current study includes uncommitted files. The bundle excludes other checkpoints, optimizer state, datasets, `.git`, virtualenvs and credentials.
3. Choose a supported Ubuntu x86-64 GPU container template, record its resolved image digest and startup configuration, and validate it locally before paying for the main run. Template selection controls the image and connection method. [Vast templates](https://docs.vast.ai/guides/instances/choosing/templates).
4. Use SSH transfer; verify file hashes after extraction. Fetch public models/assets only at the pinned revisions. The training dataset is unnecessary for inference replication. Keep the Vast management key on the controlling machine, outside the repository and transfer bundle. [Vast data movement](https://docs.vast.ai/guides/instances/storage/data-movement).

```bash
# After the local protocol is complete:
python -m evaluation.portability check
python -m evaluation.portability seal --archive outputs/cloud/evaluation-transfer.tar \
  --manifest outputs/cloud/transfer-manifest.json
# Record the printed archive/manifest hashes; transfer the archive over SSH.

# In the extracted project on a fresh Ubuntu instance:
bash scripts/cloud_setup.sh --install
source .venv/bin/activate
python -m evaluation.portability verify --manifest cloud-transfer-manifest.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python -m evaluation.hardware --label cloud-GPU-NAME --output outputs/cloud/hardware.json
python -m evaluation.prepare baseline qwen smolvlm world_model simulator_assets
python - <<'PY'
from evaluation.common import read_json
from evaluation.libero_setup import manifest
assert manifest() == read_json('studies/evaluation/initial_states.json')
PY
export CSI_CHECKPOINT="$(python -c 'import json; print(json.load(open("studies/evaluation/selection.json"))["checkpoint"]["artifact"])')"
```

The bootstrap pins uv 0.12.0, Python 3.12.11, and `uv.lock`; its default `--check` mode installs nothing. It checks package versions and CUDA/BF16 discovery, **not policy execution**. Ubuntu installation and rented-GPU execution remain untested until the future deployment. Render the LaTeX paper locally after exporting results; the cloud setup avoids installing a TeX distribution.

## Matched protocol

| Stage | Procedure / acceptance |
| --- | --- |
| Local container bridge | After the original study, rerun E2 for all four fixed checkpoints on the 3090 under the same sealed container, package versions and 8-thread settings intended for the cloud. Preserve original native-host results separately. |
| Cloud preflight | Sequentially run finite-action checks, 25-prediction timing pilots and one development episode per task/configuration. Use the **selected trained checkpoint** for S500. Verify EGL rendering, CUDA kernels and native-vs-neural equivalence before full measurements. |
| C1: efficiency | B16/Q8/Q4/S500, batch 1, identical observation-bank hash; 20 warm-ups and 500 predictions × 3 repetitions for both processor-inclusive and neural-only timing. Use the frozen runners, synchronized CUDA, and unchanged precision/quantization/TF32/attention settings. No concurrent GPU jobs. |
| C2: task replication | Target all four configurations on the same 50 reserved states/task and seeds: 500 episodes/configuration. Any smaller budget must be fixed **before** cloud outcomes are observed, and the identical subset must be extracted from local results. Never substitute development states for final states. |
| Operational evidence | Capture hardware before/after; sample GPU temperature, utilization, clocks and power during runs. Record setup/download time, measured runtime, failures, actual charges and bytes transferred. Keep samples separate from framework peak-memory metrics. |
| Closeout | Export raw CSVs/JSON, hashes, logs and figures; verify the download before destroying the rented instance. Stopping and destroying have different storage implications. [Vast instance lifecycle](https://docs.vast.ai/guides/instances/manage-instances). |

Example commands (repeat each applicable command for all four variants):

```bash
python -m evaluation.run benchmark --variant S500 --checkpoint "$CSI_CHECKPOINT" --predictions 25 --repetitions 1
python -m evaluation.run rollout --variant S500 --checkpoint "$CSI_CHECKPOINT" --phase development --episodes 1
python -m evaluation.run benchmark --variant S500 --checkpoint "$CSI_CHECKPOINT"
python -m evaluation.core_benchmark --variant S500 --checkpoint "$CSI_CHECKPOINT"
python -m evaluation.run rollout --variant S500 --checkpoint "$CSI_CHECKPOINT" --phase final --episodes 50
```

New-architecture optimized precision or kernels (e.g. FP8/FP4) would be a **separate ablation**, not a replacement for this matched comparison. Bitwise equality across GPUs is not assumed; preserve seeds, log numerical differences and measure task outcomes. If the locked stack fails, record an unsupported configuration; validate any revised stack on **both** machines in a separately identified experiment before comparing it.

## Analysis and deliverables

Analyze each machine separately with `evaluation.analysis`; it intentionally rejects mixed hardware. A later comparison must pair identical variant/checkpoint, bank, state/seed and scientific-source hashes across the two per-machine summaries. Do not weaken the existing same-hardware checks.

- Extend F1/F4 with paired cross-machine success differences and task-stratified 95% bootstrap intervals; retain the same episode denominators.
- Extend F2/F3 with per-GPU median/p95 pipeline and neural latency, within-GPU speedup versus B16, cross-GPU speedup for each fixed variant, and allocated/reserved VRAM.
- Add one compact table of CPU/GPU/driver/container/power/PCIe differences and measured cost per 1,000 chunk predictions. Show billed setup/storage/transfer overhead separately; no invented cost estimates in result plots.
- Reuse F5: weights and the adaptation history are unchanged, so no new training curve is implied.

Interpret this as a **hardware-platform comparison**, not a pure causal architecture experiment: GPU size/bandwidth, CPU, PCIe, clocks, cooling, driver and container allocation can differ. Neural-only timing reduces host preprocessing effects; the local container bridge separates packaging changes from the new machine as far as practical.
