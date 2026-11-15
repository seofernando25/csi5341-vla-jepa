# Node iter_000-b0-a2

## Change relative to parent (iter_000-b0-a1)

The parent compresses text+vision to groupwise int3 and the action head to int4, scoring
0.846808 with 8/10 episodes.  This node keeps the parent's structural changes (drop the
training-only world model, compact the token embedding, drop the unused LM head) and makes
two changes:

1. **Action head: groupwise int3 instead of int4.**  The action head is 155 M parameters,
   so moving it from 4 to 3 bits removes another ~35 MB.  It is the only precision change
   in the deployed variant; the text decoder and vision encoder keep the parent's int3
   recipe.  The action head is the most sensitive part, so this was smoke-tested rather
   than combined with any text/vision change.
2. **Exact zero-point storage.**  For the int3/int2 unpacking paths, the per-group zero
   points are small integers (0..7 / 0..3), so they are stored as uint8 instead of bf16
   and converted back to bf16 in `forward`.  This is bit-exact: the dequantized weights
   are identical, but the stored metadata is halved.  It removes ~15 MB.

`solution.py` also contains a real 2-bit packing path (`PackedInt2`) and per-linear
overrides (`TEXT_INT2_SUFFIXES`, `TEXT_INT2_LAYERS`, `VISION_INT2_BLOCKS`) for the
mixed-precision experiments.  The deployed constants leave all of those overrides empty,
so the deployed model is text int3 / vision int3 / action int3.

All replacements happen in place, so the original storages are freed; no compressed copy
is kept.  `compress` performs no file I/O and writes no files.

## Last check output

`python tools/check.py` on this workspace:

```
status: ok
memory removed: 85.6% (resident 796742630 of baseline 5532276502 bytes)
action error (max normalized difference): 0.207031
latency ms: median 447.26, p95 493.11
```

Size reduction: `1 - 796742630 / 5532276502 = 0.85598`.

## Smoke rollout (the evaluator's 10-episode phase)

The action-head-int3 variant with bf16 zero points (the only difference from this final
workspace is the exact uint8 zero-point storage) was smoke-tested:

```
status: ok, successes 8/10
per task: 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:1
```

The two failures are tasks 0 and 8, exactly the two the untouched policy and the parent
also fail; every task the parent passed still passes.  The uint8 zero-point change is
bit-exact, so the final workspace has the same forward outputs and the same 8/10 pattern.

## Experiments tried and rejected

- **Vision encoder int2 (all blocks):** 85.6% removed, action error 0.442; rejected.
- **Text `down_proj` int2 (all layers):** 85.5% removed, action error 0.302.  Smoke
  rollout failed tasks 1, 4 and 8 (7/10 at best), so it was rejected.  This is the key
  negative result: even 2-bit on only the MLP down-projections is enough to flip a task.
- **Text MLP int2 (gate/up/down):** 87.1% removed, action error 0.663; rejected.
- **Text attention int2 (q/k/v/o):** 85.5% removed, action error 0.541; rejected.
- **All text int2:** 87.9% removed, action error 0.980; rejected.

## What I would try next

- A per-group 4-centroid codebook for int2 (instead of the affine 4-level quantizer)
  should reduce the 2-bit error substantially.  The smoke result above shows the current
  affine int2 is just over the quality edge even on `down_proj`; a codebook or GPTQ-style
  error compensation is the main remaining route to 87%+ without structural pruning.
- A diagonal-Hessian-weighted int3 quantizer could improve the text decoder enough to
  make a few low-influence layers safe at 2 bits, but each candidate needs a smoke
  rollout because the 8/10 gate has no margin.
- The exact zero-point trick could also be applied to the int4 action head by storing
  zero points separately and rebuilding the tinygemm `scales_zeros` buffer per forward;
  the saving is small (~2.4 MB) and the extra per-forward allocation is not worth it here.
