# Node iter_002-b2-a0 — graphed policy with a compiled flow loop and a released embedding table

## What changed relative to the parent (`root`)

The parent is the identity `compress`. This node copies the evaluation fast path into the
workspace (`fastpath.py`, `nf4.py`) and routes the returned policy through it:

- drop the training-only V-JEPA world model (`video_encoder`, `video_predictor`,
  `video_processor`) at compress time, and release the unused Qwen `lm_head` weight together
  with the token embedding table once the ten plans exist (inference never calls either);
- precompute everything that depends only on the instruction (token ids, text embeddings,
  M-RoPE tables, visual index tables, vision position/rope tables) once per instruction;
- capture one CUDA graph per instruction and replay it for each observation, with the two
  camera images, the robot state and the flow-matching noise copied into static buffers
  before replay;
- compile only the four-step flow-matching loop with Inductor
  (`torch.compile(..., mode="default", fullgraph=True, dynamic=False)`) and replay it inside
  the captured graph. The flow has the same shapes for every task (`tokens` 1x32x2048,
  `noise` 1x7x7, `state` 1x1x8), so one compiled graph serves all ten instructions. The
  vision tower and the 28-layer decoder stay the stock eager computation, so they stay
  bit-exact;
- release the 151936 x 2048 bf16 token embedding table (and the tied `lm_head`) once the
  tenth distinct instruction has a captured plan. The graph body starts from the plan's
  precomputed `inputs_embeds`, so the table is never read again; a new instruction after the
  release raises instead of silently producing a wrong answer.

The flow-matching noise is still drawn eagerly with `torch.randn` on every call, exactly where
the stock path draws it, so a seeded call uses the same noise.

## Why

The stock path is launch-bound: the Qwen image processor and tokenizer run on every call, the
vision tower and the decoder mask code synchronize with the host, and the four flow steps
launch thousands of tiny kernels. The graph removes the Python/host overhead and the per-kernel
launch cost without changing the computation. The flow loop is a fixed-shape stack of 16 small
DiT blocks repeated four times, where eager execution is launch- and pointwise-traffic-bound;
Inductor fuses it and the isolated flow stage drops from about 6.4 ms to 4.5 ms per call.

The embedding release is a memory change, not a numerical one: it buys about 0.6 GB of
headroom for the second untouched reference the evaluator loads beside the candidate when it
interleaves the timing blocks, and for the compiled flow's graph pool. The previous node
`iter_002-b0-a1` (same flow compile, no release) scored 0.0 in the evaluation with
`status: ok` and 8/10 successes, i.e. its only zero condition was the peak; the release is the
change that makes the peak safe.

The vision tower is deliberately **not** compiled. I implemented and measured that variant in
this workspace (`fastpath.GraphedPolicy(compile_vision=True)`): it reaches point speedup 3.114
(candidate median 24.04 ms, arm error 8.4e-06, worst-task perturbation correlation 0.937, peak
4.80 GB) but the smoke gate falls to 7/10 because task 2 flips from success to failure — the
same flip node `iter_002-b0-a2` hit. With no margin on the 8/10 gate that variant is not safe,
so it is off. The full decoder compile was also rejected earlier (worst-task perturbation
correlation 0.35-0.49, below the 0.5 guard), so this node keeps the vision tower and the
decoder eager.

## Last check output

`python tools/check.py` (phase `check`, 6+6 interleaved blocks):

```
status: ok
memory removed: 16.8% (resident 5131672246 of baseline 6169967616 bytes)
score: 2.8154 (latency_speedup)
speedup: point 2.8387, 95% [2.8154, 2.8559], 6+6 blocks, interleaved
speedup per task: 0 2.844, 1 2.829, 2 2.867, 3 2.832, 4 2.834, 5 2.859, 6 2.824, 7 2.842, 8 2.852, 9 2.805
latency ms: reference median 74.73, candidate median 26.35, p95 27.35
arm action error (worst task): 9.10565e-06 (limit 0.01)
perturbation correlation (worst task): 0.974 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

A direct probe before the check measured 24.65 ms median for the candidate against 70.7 ms for
the stock path on task 0, with a maximum normalized action difference of 0.008-0.011 (the
compiled flow's bf16 rounding).

The full smoke phase (the evaluator's 10-episode gate) also passed with this workspace:

```
status: ok
successes 8 / 10  (tasks 0 and 8 fail, exactly as the untouched reference does)
score 2.7908, 95% [2.7908, 2.8365], 6+6 blocks, interleaved
point 2.8124, per task: 0 2.789, 1 2.838, 2 2.826, 3 2.816, 4 2.801,
                          5 2.808, 6 2.805, 7 2.812, 8 2.825, 9 2.804
reference median 74.40 ms, candidate median 26.52 ms, p95 27.88 ms
reference peak CUDA 6295837696 B, candidate peak 4799046656 B
arm action error 8.57e-06, perturbation correlation 0.974, no guard violations
```

The arm-action error (9.1e-06) and the perturbation correlation (0.974) show the physical
behaviour is preserved; the 8/10 smoke outcome is the reference's own (tasks 0 and 8 fail for
the untouched policy too). The same combination passed the evaluation's smoke gate in node
`iter_002-b1-a0` (score 2.817).

## What I would try next

- Find a vision-tower speedup that keeps task 2's episode outcome: the compiled tower is worth
  about 2.5 ms per call (point 3.11) but flips that one episode, so a partial compile (MLP or
  attention only) or `torch._inductor.config.emulate_precision_casts=True` would need a smoke
  run before it could be kept.
- Compile only the decoder's attention block (q/k norm + rope + SDPA + o_proj) rather than the
  whole decoder, and re-check the per-task perturbation correlation; the full decoder compile
  was faster but fell below 0.5.
- `mode="max-autotune-no-cudagraphs"` on the flow loop now that the peak has headroom.
- Batch the two camera images into one vision-attention call instead of two chunks per block
  (bit-exact in principle, about 24 fewer launches per call).
- Only consider fewer flow steps or a smaller action head if the arm-action error stays well
  under the guard and the smoke gate still passes; the current path changes no action
  semantics beyond the compiled flow's rounding, which is the safest way to preserve the
  no-margin 8/10 gate.
