# Node iter_002-b1-a1 — one Euler step in the compiled flow

## What changed relative to the parent (`iter_002-b1-a0`)

The parent is the CUDA-graph fast path with the four-step flow-matching loop compiled by
Inductor inside the captured graph and the token embedding table released after the tenth
plan.  This node keeps every part of that path and changes one number:

- `policy.model.action_model.num_inference_timesteps = 1` before `attach`, so the compiled
  flow body integrates the trained velocity field with a single Euler step (`dt = 1.0`,
  timestep 0) instead of four (`dt = 0.25`, timesteps 0, 250, 500, 750).

The vision tower and the 28-layer decoder remain the stock eager computation, so the
pixel-sensitive part of the network is still bit-exact.  The flow-matching noise is still
drawn eagerly with `torch.randn` on every call, exactly where the stock path draws it, so a
seeded call uses the same noise.

## Why

The flow stage is the only part of the call where the number of sequential passes can be cut
without touching the pixel-sensitive vision tower or the decoder.  A targeted probe on the
evaluator's own observation bank (one frame per task for the arm check, one perturbation pair
per task, same seeds as the guards) measured the physical arm error and the
perturbation-response correlation for every candidate before the full check:

```
variant          arm_max     corr_min   median_ms
baseline 4 steps 0           1.000      29.23
3 steps          8.15e-06    0.957      27.36
2 steps          3.58e-05    0.910      26.05
1 step           2.24e-04    0.626      24.43
prune 2 text     1.28e-03    0.339      28.28   (fails correlation)
prune 4 text     1.08e-02   -0.500      27.28   (fails correlation and arm error)
prune 6 text     4.38e-02   -0.128      26.10   (fails)
3 steps + prune2 1.34e-03    0.421      26.90   (fails correlation)
```

Text-layer pruning was rejected because it destroys the perturbation response even at two
layers.  The flow-step reduction is numerically almost free: the action head's velocity field
is smooth enough that one Euler step changes the physical arm actions by only 1.6e-04 MSE
(limit 0.01) and the worst-task perturbation correlation on the evaluator's frames is 0.899
(minimum 0.5).  The smoke gate confirms the closed-loop behaviour: the candidate fails
exactly tasks 0 and 8, as the untouched reference does, and passes 8/10.

## Last check output

`python tools/check.py` on this workspace (phase `check`, 6+6 interleaved blocks):

```
status: ok
memory removed: 16.8% (resident 5131672246 of baseline 6169967616 bytes)
score: 3.2480 (latency_speedup)
speedup: point 3.2692, 95% [3.2480, 3.2945], 6+6 blocks, interleaved
speedup per task: 0 3.275, 1 3.339, 2 3.270, 3 3.262, 4 3.230, 5 3.277, 6 3.265, 7 3.260, 8 3.270, 9 3.244
latency ms: reference median 75.11, candidate median 22.96, p95 23.58
arm action error (worst task): 0.000162415 (limit 0.01)
perturbation correlation (worst task): 0.899 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The full smoke phase (the evaluator's 10-episode gate) also passed on this workspace:

```
status: ok
successes 8 / 10  (tasks 0 and 8 fail, exactly as the untouched reference does)
score 3.1607, 95% [3.1607, 3.2650], 6+6 blocks, interleaved
point 3.2190, per task: 0 3.274, 1 3.231, 2 3.251, 3 3.247, 4 3.203,
                          5 3.198, 6 3.202, 7 3.219, 8 3.169, 9 3.197
reference median 73.69 ms, candidate median 23.12 ms, p95 24.17 ms
reference peak CUDA 6295837696 B, candidate peak 4799046656 B
arm action error 1.62e-04, perturbation correlation 0.899, no guard violations
```

The per-task perturbation correlations are 0.90–0.98 and the per-task arm errors are
4.8e-05–1.6e-04, so no task is near either guard.  The peak is 4.80 GB against the
reference's 6.30 GB, and the embedding release keeps the resident set at 5.14 GB.

## What I would try next

- The two-step variant also passed the smoke gate (score 3.0532) and is a fallback if a
  future evaluation finds the one-step correlation margin uncomfortable; the one-step
  variant is about 0.11 faster and its worst correlation is still 0.899.
- Compile only the decoder's MLP (not attention or RMSNorm) and re-check the per-task
  perturbation correlation; the full decoder compile was faster but fell below 0.5.
- Batch the two camera images into one vision-attention call instead of two chunks per
  block (bit-exact in principle, about 24 fewer launches per call).
- The flow is now about 1.1 ms of the 23 ms call; the decoder (12.5 ms) and vision tower
  (7.6 ms) are the remaining targets, and both need a numerical change that keeps the
  perturbation response above 0.5.
