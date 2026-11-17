# Proposal: queue-preserving fast `select_action` on the compiled CUDA graph

## Change relative to the parent (`iter_001-b0-a1`)

The parent already replays one compiled CUDA graph per instruction with two flow steps and scores
3.939648. This node keeps that execution engine and removes the remaining host overhead in the
timed observation-to-action call:

1. `policy.select_action` is replaced by the same action-queue logic without the per-call
   `self.eval()` recursive module walk and without `populate_queues` (which only has the `ACTION`
   key and is called with `exclude_keys=[ACTION]`, so it is a no-op). The queue is preserved: an
   episode still plans one chunk and executes its `n_action_steps` actions, so episode behaviour
   is bit-identical to the parent's.
2. Everything else is unchanged: the inference-dead V-JEPA world model and the tied vocabulary
   head are dropped, the graphed body is compiled with `torch.compile`, and the flow head runs
   two steps.

## Why it should pass the guards

The change is host-side only and does not alter the numbers the policy returns. The stock
`select_action` calls `self.eval()` on every call; the policy is already in eval mode (the
evaluator loads it with `.eval()`), so the walk is a no-op. `populate_queues` cannot add anything
because `_queues` only holds `ACTION`, which is excluded. The action queue is kept, so the
episode rollout still executes seven actions per planned chunk exactly as the parent did.

Local timing on a fixed bank frame, same graph and same flow steps:

| call | median wall | median GPU |
|---|---|---|
| stock `select_action` | 18.49 ms | 18.51 ms |
| `predict_action_chunk` | 17.08 ms | 17.10 ms |
| **fast `select_action` (this node)** | **17.10 ms** | **17.11 ms** |

The 1.4 ms difference is the `self.eval()` walk plus queue bookkeeping, not GPU work. The graph
body itself is unchanged, so the parent's guard figures carry over: worst-task arm action error
4.25e-05 (limit 0.01), perturbation correlation 0.928 (minimum 0.5), 8/10 smoke successes, peak
CUDA 5.42 GB against the reference's 6.30 GB.

I also measured more aggressive variants, but did not use them because the smoke gate has no
margin. One flow step gives worst-task arm MSE 2.0e-4 and correlation 0.884; one flow step plus a
one-step action prompt gives arm MSE 1.2e-3 and correlation 0.774; skipping action-head layers
3-11 with two flow steps gives arm MSE 5.3e-3 and correlation 0.671. Block skipping in the text or
vision decoder destroys the perturbation response (text 3-5: correlation 0.401; vision 16-22:
0.199), so it stays out. The chosen node keeps the parent's proven flow-2 arithmetic and only
removes host overhead.

## Last check output

`/tmp/venv/bin/python tools/check.py` (phase `check`, task 0 episode, all ten tasks timed):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 4.4326 (latency_speedup)
speedup: point 4.4511, 95% [4.4326, 4.4760], 6+6 blocks, interleaved
speedup per task: 0 4.459, 1 4.519, 2 4.446, 3 4.449, 4 4.438, 5 4.447, 6 4.439, 7 4.488, 8 4.446, 9 4.382
latency ms: reference median 75.05, candidate median 17.03, p95 17.59
arm action error (worst task): 4.04965e-05 (limit 0.01)
perturbation correlation (worst task): 0.914 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The parent's check scored 4.0736 at 18.45 ms; this node's 17.03 ms and 4.4326 are the same
graph body with the host-side `select_action` overhead removed. The arm error and perturbation
correlation are unchanged within run-to-run noise, as expected for a host-only change.

## What I would try next

- Validate one flow step (arm MSE 2.0e-4, correlation 0.884) with a full ten-episode smoke run;
  it saves about 0.4 ms of graph time and is the next safe increment.
- Move the 256->224 area resize and the raw image copy into the graph's static buffers, removing
  the host-side `_prepare_model_inputs` path entirely (about 0.1 ms).
- Try `max-autotune` GEMM selection for the decoder MLP shapes once a warm compile cache is
  available; the profile is dominated by cuBLAS/cutlass GEMMs at about 83% of the 3090's bf16
  peak, so kernel choice is the remaining lever.
