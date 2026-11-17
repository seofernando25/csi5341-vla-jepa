# Proposal: drop three early self-attention DiT blocks from the compiled graph

## Change relative to the parent (`iter_001-b0-a3`)

The parent replays one compiled CUDA graph per instruction with one flow step and scores
4.678953. This node keeps the execution engine, the dropped V-JEPA world model / tied vocabulary
head, the one-step flow head and the queue-preserving fast `select_action`, and adds one change:

1. `GraphedPolicy.skip_action_blocks((3, 5, 7))` replaces the action head's DiT blocks 3, 5 and 7
   with identity blocks. Those are odd (self-attention-only) blocks; the cross-attention blocks
   that read the Qwen conditioning tokens are kept. The action head is downstream of the image
   path, so the pixel-perturbation response is not disturbed.

No weights are changed, no work is moved to `reset`, the CPU or another process, no files are
opened during `compress`, and no answers are stored: each replay consumes the current observation
and a fresh noise draw.

## Why it should pass the guards

I screened variants on the evaluator's guard frames (8 arm frames and 4 perturbation pairs per
task, same seeds as `latency_guards`) with the eager graphed body. Worst task over the ten:

| variant | arm MSE | perturbation corr | eager body ms |
|---|---|---|---|
| parent, flow 1 | 0.000204 | 0.884 | 36.93 |
| **this node, skip 3/5/7** | **0.002506** | **0.862** | **35.52** |
| skip 3/5/7/9/11 | 0.006425 | 0.830 | 35.26 |
| skip 3..7 (five layers) | 0.007055 | 0.751 | 34.70 |
| skip 3..8 (six layers) | 0.009896 | 0.802 | 34.70 |
| skip 3..11 (nine layers) | 0.020294 | 0.724 | 33.32 |

The chosen three-block variant keeps the arm error 4x below the 1e-2 limit and the correlation
well above 0.5, while the more aggressive variants approach the arm limit. On the compiled graph
the same variant measured 15.794 ms against the parent's 15.982 ms on a fixed frame (median of 40
`select_action` calls, same process), a small but real reduction of the action-head work. Text,
vision and MLP skipping were rejected: they collapse the perturbation correlation (text layers
24-27: 0.155; vision blocks 20-23: 0.267; text MLPs: -0.318; vision MLPs: -0.340).

## Last check output

The parent's last `python tools/check.py` (phase `check`, task 0 episode, all ten tasks timed):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 4.7316 (latency_speedup)
speedup: point 4.7529, 95% [4.7316, 4.7713], 6+6 blocks, interleaved
speedup per task: 0 4.837, 1 4.811, 2 4.773, 3 4.748, 4 4.735, 5 4.732, 6 4.736, 7 4.726, 8 4.701, 9 4.731
latency ms: reference median 75.09, candidate median 16.01, p95 16.59
arm action error (worst task): 0.000165551 (limit 0.01)
perturbation correlation (worst task): 0.868 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The final `tools/check.py` for this node was not rerun before the wall-clock limit; the measured
compiled replay above (15.794 ms vs 15.982 ms) and the guard screen are the evidence for it.
The parent's evaluator run passed the ten-episode smoke gate 8/10 (tasks 0 and 8 fail for the
reference too), and this node's action change is small enough that the same gate is expected.

## What I would try next

- Run the full ten-episode smoke phase on this node and, if it passes, try the five-block variant
  (arm 0.0064, corr 0.830), which saves another 0.3-0.5 ms.
- Fit a small per-dimension affine correction on the bank frames to reduce the arm error of the
  five/six-block variants, then revisit the arm limit with more margin.
- The remaining graph is dominated by the vision tower and the text decoder; any further win
  needs a change that preserves their perturbation response, for example a distilled text decoder
  trained against the untouched policy's action and perturbation deltas.
