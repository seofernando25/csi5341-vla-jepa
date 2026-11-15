# Node iter_000-b0-a1

## Change relative to parent (iter_000-b0-a0)

The parent dropped the training-only world model, compacted the token embedding and
quantized text+vision to groupwise int4 with AWQ-style per-channel scaling, leaving the
action head at int8.  It scored 0.795718 with 8/10 successful episodes.

This node keeps the two inference-exact structural changes and pushes the approximate
part further:

1. **Text decoder: groupwise int3 with real 3-bit packing.**  Eight consecutive input
   columns are packed into three uint8 bytes (3/8 byte per parameter) and dequantized to
   bfloat16 per forward.  The AWQ per-channel input scale and 10-point alpha search are
   kept, with 8 affine levels instead of 16.  This is the main size win: the text decoder
   is 1.41 B parameters and dominated the parent's resident bytes.
2. **Vision encoder: groupwise int3** (same packing) instead of int4.  The vision tower
   is 407 M parameters.
3. **Action head: groupwise int4** instead of int8.  The DiT action head is 155 M
   parameters; int4 halves its storage.  It uses the same AWQ scaling and PyTorch's
   tinygemm int4 kernel.
4. Norms, the compact embedding and the timestep encoder stay bfloat16; any Linear
   without calibration data or with an input width not divisible by 128 falls back to
   weight-only int8.

All replacements happen in place, so the original storages are freed; no compressed
copy is kept.  `compress` performs no file I/O and writes no files.

## Last check output

`python tools/check.py` on the final workspace:

```
status: ok
memory removed: 84.7% (resident 847500222 of baseline 5532276502 bytes)
action error (max normalized difference): 0.201172
latency ms: median 317.66, p95 366.46
```

## Smoke rollout (the evaluator's 10-episode phase)

`evaluation.compression.study solution --workspace /tmp/work --phase smoke`:

```
status: ok, successes 8/10
resident 847,500,222 bytes, size_reduction 0.8468080506
per task: 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:1
```

The two failures are tasks 0 and 8, exactly the two the untouched policy and the parent
also fail; every task the parent passed still passes.

## Experiments tried and rejected

- Text int3 + vision int4 + action int4: 83.8% removed, 8/10 (same per-task pattern).
- Text int3 + vision int3 + action int4: 84.7% removed, 8/10 (final choice).
- Adding removal of the four lowest-influence text blocks (layers 3-6, calibration
  residual influence ~1e-4) reached 86.2% but only 4/10 episodes; reverted.

## What I would try next

- The text decoder is still ~0.57 GB at int3.  A GPTQ-style Hessian-aware int3 or a
  mixed int2/int3 allocation could remove another 0.1-0.2 GB, but it needs real
  calibration activations (only per-channel input energy is frozen here) and a smoke
  rollout per candidate because the 8/10 gate has no margin.
- Pruning text blocks is not viable at this quality level: skipping layers 3-6 lost four
  episodes even though their calibration residual influence was tiny.
