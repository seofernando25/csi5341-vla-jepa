# Proposal: bit-exact CUDA graph with cross-attention K/V reuse in the flow head

## Change relative to the parent

The parent (`iter_000-b0-a1`, score 2.787) routes `predict_action_chunk` through the
bit-exact CUDA-graph executor in `graphed.py`, drops the training-only V-JEPA branch and
removes the per-call `self.eval()` tree walk from `select_action`. This node keeps that
executor and its exact numerics, and adds one bit-exact optimisation:

- **The flow head's cross-attention K/V projections are computed once per call instead of
  once per flow step.** The DiT cross-attends to the 32 embodied action tokens produced by
  the decoder. Those tokens are constant across the four flow steps, so `to_k(tokens)` and
  `to_v(tokens)` are identical in every step. `GraphedPolicy` now installs a
  `_CachedCrossAttention` processor on the eight cross-attention blocks; `_flow` computes
  the two projections once and the processor reuses them. This removes 48 small GEMMs per
  call (8 blocks x 2 projections x 3 extra steps) and changes no value: the projections
  are deterministic and the cached tensors are the same values the stock processor would
  recompute. `solution.py` enables it with `attach(policy, cache_kv=True)`.

The rest of the executor is unchanged: the same vision tower, Qwen decoder, DeepStack
injections and four-step flow head; the same static image/state/noise/output buffers; the
same stock noise draw (`torch.randn` with the same shape, dtype and device before replay);
the same lean `select_action` queue semantics; one graph per prompt length, captured for
the ten suite instructions in `compress`, before the evaluator's warm-up and timed blocks.

`GraphedPolicy` also accepts `use_compile`, `emulate` and `compile_flow` options and keeps
the `_CachedCrossAttention` implementation in the file. They are off in the submitted
path; the development probes (`dev_compile.py`, `dev_profile.py`, `dev_kernels.py`,
`dev_verify.py`) are not imported by `solution.py`.

## Why it should be faster and why it is safe

The flow head is the most launch-bound stage of the graph: its 16 DiT blocks run four
times with 40-token sequences, and the kernel profile showed about 490 small cutlass GEMMs
(~7.9 ms of a ~31 ms replay on the shared GPU). The cross-attention K/V projections are a
pure function of the conditioning tokens, which do not change between steps, so hoisting
them out of the step loop is bit-exact and removes 48 GEMM launches per call.

The 8/10 gate has no margin (the reference itself fails tasks 0 and 8), so a numeric change
is not acceptable. The cached path was verified against the untouched policy on four bank
frames: maximum absolute action difference 0.0. The evaluator's check reports arm action
error 0 on the worst task and perturbation correlation 1.000, and the ten-episode smoke
phase reproduces the reference's exact success pattern (tasks 0 and 8 fail, all others
pass).

## Last check output

`PYTHONPATH=/tmp/repo /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 15.4% (resident 5216743424 of baseline 6169967616 bytes)
score: 2.9864 (latency_speedup)
speedup: point 3.0125, 95% [2.9864, 3.0408], 6+6 blocks, interleaved
speedup per task: 0 2.990, 1 3.032, 2 3.036, 3 3.016, 4 3.004, 5 2.972, 6 3.018, 7 3.042, 8 3.007, 9 3.009
latency ms: reference median 79.76, candidate median 26.17, p95 29.34
arm action error (worst task): 0 (limit 0.01)
perturbation correlation (worst task): 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The evaluator's ten-episode smoke phase (`--phase smoke`, the same development states and
seeds the node is scored on) also completed:

```
status ok, score 3.0596, successes 8/10, violations []
per task: 0 fail, 1-7 pass, 8 fail, 9 pass
candidate median 26.57 ms, p95 27.90 ms
arm action error 0.0, perturbation correlation 1.000
peak CUDA 5.343 GB against reference 6.296 GB
```

The failure pattern is exactly the untouched policy's, so the bit-exact path adds no new
episode failure. The parent's check on the same harness was 2.8312 with a candidate median
of 26.52 ms; the new check is 2.9864 with 26.17 ms, and the smoke phase scores 3.0596.

## What I tried that did not work

- Full-body `torch.compile` inside the graph: 141 s to compile, maximum normalized action
  difference 0.0156, worst physical arm MSE 7.85e-6. `emulate_precision_casts=True` did
  not reduce the drift, so the compiled path is not bit-exact. Node `iter_000-b2-a0` took
  this path and failed the smoke gate at 7/10 (task 2), so it was rejected here.
- Flow-only `torch.compile`: 33 s to compile, arm MSE 7.35e-6, and the candidate median was
  slower than the bit-exact graph (~31 ms against ~26.5 ms) because the tiny flow GEMMs are
  faster through cuBLAS than through the compiled kernels. This rules out the flow-only
  compile as a safe win.
- The first cached-K/V implementation divided the already-reshaped key by the head count
  in the processor; the shape error was found by `dev_verify.py` and fixed by taking the
  head dimension from the query projection. The final path is bit-exact on the bank.

## What I would try next

- Cache the state-encoder output across the four flow steps (the state is constant within
  a call); it removes six more small GEMMs and is bit-exact.
- Compile only the vision tower (the largest launch-bound stage) and validate the full
  ten-episode smoke gate in the evaluator's own phase before submitting; a vision-only
  drift may be small enough for the no-margin gate, unlike the full-body compile.
- If the flow head is to be compiled, use a fixed kernel selection and require a
  ten-episode gate pass in the evaluator's smoke phase, not just a local run.
