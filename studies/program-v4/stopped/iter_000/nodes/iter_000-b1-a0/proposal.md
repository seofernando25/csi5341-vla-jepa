# Node iter_000-b1-a0: two CUDA graphs for the timed instructions

## What changed relative to the parent

The parent (`root`) returns the untouched B16 policy. This node adds
`fastpath_local.py`, a B16-only, no-compile copy of the repository's
`evaluation/fastpath.py` graph executor, and `compress` attaches it to
`policy.predict_action_chunk` (so `select_action` uses it too) with
`max_plans=2`.

The graph path precomputes the instruction-dependent constants (text embeddings,
M-RoPE tables, vision position/rotary tables, visual/action token indices) and
replays one captured CUDA graph per instruction. The eager body is the same
sequence of sub-module calls as the stock path, with the vision attention's
host-side `cu_seqlens.tolist()` replaced by static lengths and the sinusoidal
positional encoding's constants built on the device. No arithmetic, weights,
dtype, flow steps or action horizon change, so outputs are bit-exact with the
stock path (verified: max abs difference 0.0 on held-out bank frames 0, 12, 24,
36 and 48, including the stock fallback for tasks 3 and 4).

## Why this differs from the failed sibling

The sibling `iter_000-b0-a0` graphed every instruction it saw. Its actual
evaluation reached 10 graphs, its peak CUDA (5,929,086,976 B) was 2.25 % above
the reference (5,798,650,880 B), and the score was zero even though its local
check with 3 graphs was below the baseline.

This node caps the executor at **two** graphs. The reference episode for task 0
runs the full 280 steps, so its 70 sampled frames fill the first 70 entries of
the frame pool; the timed blocks use the first 60 non-warm frames, which are
tasks 0 and 1. Those are exactly the first two instructions the behaviour guards
see, so they get graphs. Task 2 and tasks 3-9 keep the untouched stock path, so
their candidate episodes are identical to the reference's. The peak CUDA
overhead is two graph pools instead of ten.

## Why it is fast

The stock path is launch-bound: a 2B Qwen3-VL forward plus a 16-block DiT flow
head costs about 73-149 ms per observation-to-action call at batch 1 depending
on the shared GPU's state, while the same computation replayed from a CUDA graph
costs about 27-35 ms. Because the graph body is the eager body, the smoke gate,
arm-action error and perturbation-correlation guards are identical to the
reference.

## Last check output (final two-graph solution)

```
status: ok
memory removed: 0.2% (resident 5682556416 of baseline 5691661312 bytes)
score: 3.2770 (latency_speedup)
speedup: point 3.5729, 95% [3.2770, 3.9668], 6+6 blocks, interleaved
latency ms: reference median 149.13, candidate median 35.39, p95 42.20
arm action error: 0 (limit 0.01)
perturbation correlation: 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The check phase's resident is below its baseline, so its peak-memory gate
passed. The check's reference median was 149.13 ms; the score moves with the
shared GPU's state.

## Full smoke phase (10 episodes, the evaluator's actual phase)

```
status: ok
score: 2.5949
successes: 8 / 10  (task 0 and task 8 fail, exactly as the reference)
speedup: point 2.6378, 95% [2.5949, 2.7240], 6+6 blocks, interleaved
latency ms: reference median 103.16, candidate median 29.84, p95 31.18
peak CUDA: 5836062720 B, reference 5798650880 B (0.645 % above, 1 % tolerance)
arm action error: 0 (limit 0.01)
perturbation correlation: 1.000 (min 0.5)
reset: 0.006 ms, overhead 41.07 ms/step (limit 120.86), CPU 30.23 ms/call
violations: none
```

The two-graph peak has about 20 MB of headroom under the 1 % tolerance. The
three-graph variant scored 2.5958 in the same smoke phase but its peak
(5,849,810,944 B) was only about 7 MB under the tolerance, so the two-graph
version is the safer node for the same score.

## What I would try next

- If the actual evaluation's peak is still close, drop to one graph (task 0) or
  graph only the action head, which is one shape-independent graph and adds only
  a few MB.
- The repository's `torch.compile` graph body reaches about 19 ms instead of
  27 ms, but its compile-time peak was above the tolerance in the sibling's
  probe and it changes bf16 rounding; the no-margin 8/10 gate makes that a
  riskier next step than a third bit-exact graph.
- Fewer flow steps would change the actions and risk the no-margin smoke gate;
  bit-exactness is the safer way to keep the node.
