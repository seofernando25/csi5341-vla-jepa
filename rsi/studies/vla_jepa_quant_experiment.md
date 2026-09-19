# Experiment: Quantized VLA-JEPA Benchmark & Initialization (`vrfai/vla-jepa-libero`)

## Overview

This experiment branch (`exp-vla-jepa-quant`) adapts the LeRobot VLA-JEPA + LFM2.5-VL research framework to support the quantized GGUF release of VLA-JEPA ([`vrfai/vla-jepa-libero`](https://huggingface.co/vrfai/vla-jepa-libero)) as:
1. An alternative reference baseline for closed-loop confirmation and evaluation on the LIBERO-Spatial benchmark.
2. A compatible initialization source for the DiT flow-matching action head via hybrid Safetensors/GGUF weight loading.

## Model Details

* **Base Model**: `lerobot/VLA-JEPA-LIBERO` converted to GGUF format (`vla-jepa.gguf`, 4.25 GiB).
* **Backbone**: Qwen3-VL-2B-Instruct vision-language transformer.
* **Action Head**: DiT-B flow-matching action head (denoising a 7-step action chunk over 4 flow-matching steps).
* **World Model**: Dropped in the GGUF conversion (as documented by upstream, the V-JEPA predictor is off the inference critical path).

## Changes Made

1. **`src/lerobot_policy_vla_jepa_lfm/modeling_vla_jepa_lfm.py`**:
   * Enhanced `_load_compatible_vla_jepa_weights` to detect and load `.gguf` format weights using `gguf.GGUFReader`.
   * Automatically resolves Hugging Face Hub repos providing `.gguf` bundles.
   * Handles dropped components (e.g. world model) gracefully when transferring compatible action-head weights.

2. **`src/lerobot_policy_vla_jepa_lfm/configuration_vla_jepa_lfm.py`**:
   * Updated `init_from_vla_jepa` default target to `vrfai/vla-jepa-libero`.

3. **`rsi/confirmation.json`**:
   * Updated `qwen_checkpoint` to `vrfai/vla-jepa-libero` to serve as the evaluation baseline template for edge/quantized benchmarks.

4. **`pyproject.toml`**:
   * Added `gguf>=0.10.0` to project dependencies.

## Hardware & Resource Considerations

* **Local Compute Constraint (Original)**: The original author's host was constrained by an NVIDIA GeForce GTX 1650 with 4 GB VRAM.
* **Current Evaluation Hardware**: Validated on host GPU **NVIDIA GeForce RTX 5060 Ti (16,311 MiB / 16 GB VRAM)** with >15 GiB free VRAM, eliminating the 4 GB OOM bottleneck.
* **Memory Footprint**: `vla-jepa.gguf` occupies 4.25 GiB in BF16 weights alone.

## Validation & Updated Metrics

### 1. Test Suite Verification
* **Pytest Suite**: **64 / 64 tests passing** (`test_plugin.py`: 6, `test_rsi.py`: 17, `test_rsi_v2.py`: 17, `test_rsi_v3.py`: 16, `test_rsi_v4.py`: 8) in 151.8s.
* **GGUF Unit Tests**: Added unit coverage in `tests/test_plugin.py` verifying:
  * `test_load_compatible_weights_gguf`: Successful tensor extraction and state dict loading from GGUF format via `gguf.GGUFReader`.
  * `test_load_compatible_weights_gguf_mismatch`: Strict shape mismatch detection raising `ValueError("Incompatible VLA-JEPA initialization tensors")`.
  * `test_load_compatible_weights_safetensors`: Safetensors format parity verification.
* **Plugin Registration**: Verified via `scripts/smoke_plugin.py` (`vla_jepa_lfm` -> `lerobot_policy_vla_jepa_lfm.modeling_vla_jepa_lfm.VLAJEPALFMPolicy`).
* **Harness Dry-Run**: Validated via `python -m rsi dry-run`: 8/8 attempts reserved, 7 distinct families, 6 promotions completed, 1 completed offline cycle, exactly 100 dream replay trajectories, finished cleanly with `global_outer_cap`.
* **Linting & Formatting**: Clean pass with Ruff across `src`, `rsi`, and `tests`.

### 2. GGUF Tensor Layout & Prefix Metrics
* **Remote Hub Inspection (`vrfai/vla-jepa-libero`)**:
  * File size: 4.25 GiB (`vla-jepa.gguf`).
  * Total tensors: 873 tensors.
  * Tensors present: `ah.*` (DiT-B action head), `vit.*` (Qwen3-VL vision encoder), `vlm.*` (Qwen3 language model backbone), `token_embd.weight`. World-model predictor is dropped upstream as documented.
  * Architecture prefix observation: Action head weights in `vla-jepa.gguf` utilize the `ah.` namespace (e.g., `ah.act_enc.*`, `ah.timestep_encoder.*`), compared to standard LeRobot checkpoints which expose `model.action_model.*`. The compatibility loader gracefully logs a warning and falls back to default initialization when prefixes do not match, without crashing.

