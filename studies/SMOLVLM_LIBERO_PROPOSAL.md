# SmolVLM replacement with LIBERO trained VLA JEPA weights

**Status: proposed next experiment; not started.** Replace Qwen with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` while retaining the compatible trained components of `lerobot/VLA-JEPA-LIBERO`. This stays within the original smaller-backbone project direction.

## Motivation and hypothesis

The earlier [training setup](../evaluation/train.py) copied the LIBERO policy configuration but initialized compatible action/world modules from `VLA-JEPA-Pretrain`. Consequently, it did not test retaining the LIBERO-trained controller. The [audit](audit/20261002/README.md) also identified restricted adaptation and query trainability as unresolved factors. Low validation loss did not establish successful robot control.

**Hypothesis:** retaining LIBERO-trained action/world components and learning the SmolVLM conditioning interface can recover useful closed-loop performance with a limited adaptation budget. This is untested; neither longer training nor successful transfer is guaranteed. The separate [SmolVLA result](smolvla/README.md) supports investigating compact backbones but uses a different controller and no JEPA.

## Initialization and adaptation

1. Pin the LIBERO checkpoint and SmolVLM revisions. Restore compatible LIBERO action-head and world-predictor tensors, retaining the fixed V-JEPA encoder. Replace Qwen with SmolVLM and its conditioning adapter. Export a per-module weight-transfer manifest: restored, newly initialized and incompatible tensors. Do not silently skip mismatches.
2. Verify checkpoint action normalization, camera processing, query construction and gradient connectivity before training. Preserve action/world architectures, losses, action horizon and flow-matching prediction semantics. Keep the corrected tensor/GPU processing path common across measured variants; record backbone-native processing differences.
3. **Stage A:** freeze SmolVLM, V-JEPA and the restored action/world modules. Train the adapter and any newly introduced conditioning/query parameters, explicitly recording their trainability. Measure an unadapted control before this stage.
4. **Stage B, conditional:** if Stage A remains inadequate, continue from its checkpoint and selectively fine-tune the action head; permit predictor fine-tuning as a separately recorded intervention. Keep the backbone frozen initially. Record exactly what changes between stages.

Qwen remains evaluation-only. This is a separate follow-up, not an amendment to the frozen Dream-RSI study.

## Evaluation and decision

- Freeze the training-step/GPU-hour budget, optimizer, evaluation schedule and checkpoint-selection rule before execution. Budget is currently undecided; this proposal schedules no compute or spending.
- Retain the fixed whole-trajectory split and held-out validation set. At predeclared checkpoints, measure both validation loss and development LIBERO success. Use the same development states/seeds across stages; a ten-task one-episode check is diagnostic only. Use 100 development episodes for a substantive comparison.
- Select using development task success, with validation loss as a predeclared tie-breaker. Keep the 500 final episodes reserved for the selected checkpoint. Compare against B16 on matched states, seeds, hardware and inference settings.
- Measure warmed-up median/p95 pipeline latency and peak allocated/reserved inference VRAM separately from simulator time. Retain the project target: at least 25% lower inference VRAM with at most a five-percentage-point success drop; report uncertainty.
- To attribute an initialization benefit, compare LIBERO versus Pretrain initialization with the same architecture, trainability, seed, processing and adaptation budget. Historical runs alone cannot isolate this effect; if this control is unaffordable, label the follow-up as a combined intervention.

Deliver the transfer/trainability manifests, reproducible configuration, loss and development-success curves, final per-task outcomes if evaluated, and a success/latency/VRAM comparison. Preserve prior results under their original provenance.
