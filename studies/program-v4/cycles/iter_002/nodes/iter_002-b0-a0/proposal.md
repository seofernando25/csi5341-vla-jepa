# Node iter_002-b0-a0 — CUDA-graph execution of VLA-JEPA

## What changed relative to the parent (`root`)

The parent is the identity `compress`. This node copies the evaluation fast path
(`evaluation/fastpath.py` and its `evaluation/nf4.py` helper) into the workspace and uses it
inside `compress`:

- drop the training-only V-JEPA world model (`video_encoder`, `video_predictor`,
  `video_processor`) and the unused Qwen `lm_head`, which inference never calls;
- precompute everything that depends only on the instruction (token ids, text embeddings,
  M-RoPE tables, visual index tables, vision position/rope tables) once per instruction;
- capture one CUDA graph per instruction and replay it for each observation, with the two
  camera images, the robot state and the flow-matching noise copied into static buffers
  before replay;
- draw the flow-matching noise eagerly with `torch.randn` on every call, exactly where the
  stock path draws it, so a seeded call uses the same noise.

`use_compile=False` and `dense=True` are deliberate: the graph body is the stock eager
computation, so its output equals the untouched policy bit for bit. On the 10-task
observation bank the copied path matched the stock `predict_action_chunk` exactly
(max abs difference 0.0 on every task, one frame per task plus a repeat). That makes the
equivalence, arm-action and perturbation guards exact rather than approximate, and it
preserves the reference's episode outcomes exactly.

## Why

The stock path is launch-bound: the Qwen image processor and tokenizer run on every call,
the vision tower and decoder mask code synchronize with the host, and the four flow steps
launch thousands of tiny kernels. The graph removes the Python/host overhead and the
per-kernel launch cost without changing the computation, so every task gets the same
speedup and the per-task geometric mean is not carried by one task. The candidate's GPU
time per call is 28.28 ms against the reference's 76.58 ms, so the graph is GPU-bound and
close to the model's compute limit; the 14 % memory reduction (world model and lm_head
removed) leaves headroom under the reference's peak CUDA bytes.

## Last check output

`python tools/check.py` (phase `check`, 6+6 interleaved blocks):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 2.6193 (latency_speedup)
speedup: point 2.6644, 95% [2.6193, 2.6968], 6+6 blocks, interleaved
speedup per task: 0 2.687, 1 2.677, 2 2.695, 3 2.695, 4 2.619, 5 2.646, 6 2.636, 7 2.665, 8 2.674, 9 2.651
latency ms: reference median 76.94, candidate median 28.41, p95 29.62
arm action error (worst task): 0 (limit 0.01)
perturbation correlation (worst task): 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The full smoke phase (`--phase smoke`, the evaluator's 10-episode gate) also passed:

```
status: ok
successes 8 / 10  (tasks 0 and 8 fail, exactly as the untouched reference does)
score 2.6152, 95% [2.6152, 2.7139], 6+6 blocks, interleaved
per task: 0 2.672, 1 2.674, 2 2.647, 3 2.654, 4 2.693, 5 2.616, 6 2.639, 7 2.631, 8 2.674, 9 2.636
reference median 76.56 ms, candidate median 28.26 ms
reference peak CUDA 6295837696 B, candidate peak 5421900800 B
arm action error 0.0, perturbation correlation 1.0, no guard violations
```

## What I would try next

- `use_compile=True` on the graph body (fuses kernels and changes bf16 rounding); it needs
  the arm-action-error check and a smoke run, because the 8/10 gate has no margin.
- Pre-building all ten instruction plans inside `compress` so no capture happens during the
  untimed equivalence check (a robustness change, not a timing change).
- Reducing flow steps or visual tokens only if the arm-action error stays under the guard
  and the smoke gate still passes; the current path is bit-exact, which is the safest way
  to preserve the no-margin gate.
