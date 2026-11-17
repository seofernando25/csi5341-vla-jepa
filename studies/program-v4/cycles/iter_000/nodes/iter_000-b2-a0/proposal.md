# Proposal: compiled CUDA-graph execution of the VLA-JEPA action chunk

## Change relative to the parent

The parent is the identity policy (`compress` returns the loaded B16 policy, score ~1.0). This
node keeps the CUDA-graph executor of the earlier graph nodes but adds the step they avoided:
the graph body is fused with `torch.compile` (default Inductor mode, no nested CUDA graphs).
The executor is vendored into the workspace as `fastpath.py` and imported at module load, so
`compress` opens nothing outside the workspace.

- The same vision tower, Qwen3-VL decoder, DeepStack injections and four-step flow head are
  composed without the stock path's host synchronizations (vision `.tolist()`, decoder mask
  code), then compiled and captured into one CUDA graph per prompt shape.
- `compress` drops the training-only V-JEPA video encoder, predictor and processor (inference
  never calls them), which frees their weights and leaves the graph pool a wide peak-memory
  margin.
- `compress` compiles and captures all ten suite instructions before the evaluator's warm-up,
  so no compilation and no graph capture happens during the timed blocks. Compilation takes
  about 94 s for the five distinct prompt lengths, well inside the 900 s limit. An unseen
  instruction falls back to lazy capture.
- Two bit-exact micro-optimisations are included: the image resize is skipped when the input is
  already at the target side (the stock `_prepare_model_inputs` has already resized to 224), and
  the DeepStack injection writes in place instead of cloning the decoder state.
- The stock flow-noise draw is preserved (`torch.randn` with the same shape, dtype and device
  before replay), so reproducibility, `select_action`/`predict_action_chunk` agreement and the
  perturbation response are unchanged.

The compiled path is not bit-exact: fusing the elementwise chains changes bf16 rounding. The
check measures the result against the untouched policy: maximum physical arm-action MSE
9.2e-6 (limit 0.01) and worst-task perturbation correlation 0.921 (min 0.5), so the policy is
still the same policy.

## Why it should be faster

The stock path is launch-bound: each prediction launches thousands of small kernels and
synchronizes with the host inside the vision tower and the decoder mask code. Replaying one
captured graph removes the launch overhead and the host synchronization; `torch.compile`
additionally fuses the elementwise chains (RMSNorm, residual adds, SiLU/gates, the flow head's
per-step updates) that dominate the remaining kernel count. Measured graph replay falls from
26.1 ms (CUDA graph alone) to 19.1 ms, and the evaluator's full `select_action` median falls
from 75.5 ms (reference) to 20.5 ms.

## Last check output

`PYTHONPATH=/tmp/repo /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 3.7030 (latency_speedup)
speedup: point 3.7189, 95% [3.7030, 3.7317], 6+6 blocks, interleaved
speedup per task: 0 3.739, 1 3.709, 2 3.716, 3 3.768, 4 3.717, 5 3.710, 6 3.720, 7 3.721, 8 3.674, 9 3.714
latency ms: reference median 75.47, candidate median 20.53, p95 21.13
arm action error (worst task): 9.24502e-06 (limit 0.01)
perturbation correlation (worst task): 0.921 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The check's task-0 episode behaves as the reference's does. The peak CUDA figure is inside the
reference's because the world-model branch is dropped.

I also ran the evaluator's ten-episode gate locally (`--phase smoke`, the same development
states and seeds the node is scored on):

```
status ok, score 3.5848, successes 8/10, violations []
per task: 0 fail, 1-7 pass, 8 fail, 9 pass
```

The failure pattern is exactly the untouched policy's (tasks 0 and 8), so the compiled path
reproduces the reference's 8/10 gate with no new episode failure.

## What I would try next

- If the evaluator's ten-episode gate ever shows one new failure from the small bf16 drift,
  fall back to `use_compile=False` (bit-exact CUDA graph, score ~2.7) or compile only the
  vision/decoder and leave the flow head eager, then re-check the episode count.
- If more speed is needed, compile with `mode="max-autotune"` and a longer `compress` budget,
  or precompute the flow head's cross-attention K/V once per call instead of once per flow step.
- If the peak figure is unexpectedly high in the evaluator, capture graphs for the five prompt
  lengths only (sharing static buffers) instead of one graph per instruction.
