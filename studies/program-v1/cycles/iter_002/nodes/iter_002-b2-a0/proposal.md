# Node iter_002-b2-a0: dead-module removal + 4-bit NF4 on every live tower

Parent: `root` (identity `compress`, score 0.0). Sibling `iter_002-b1-a0` reached
0.683 with an INT8 text decoder and NF4 vision/action. This node keeps the proven
dead-module removal and compact embedding, but puts the **text decoder itself** in
4-bit NF4. The decoder is the only remaining large lever: it is ~2.82 GB of the
5.53 GB baseline, ~1.41 GB at INT8 and ~0.70 GB at NF4.

## Change relative to the parent

All changes are in place on the loaded policy, so the original storages are released.

1. **Drop the V-JEPA world model** (`video_encoder`, `video_predictor`,
   `video_processor`): inference never calls it. ~967 MB freed.
2. **Drop the tied LM head** and **compact the token embedding** to the frozen
   `calibration["token_ids"]` vocabulary. Unknown ids map to a zero row instead of
   raising. ~622 MB freed.
3. **NF4 (bitsandbytes, group 64, double-quantized absmax, bf16 compute) every
   `nn.Linear`** in the text decoder, vision tower and action head. A zero-element
   bf16 anchor keeps the action head's `TimestepEncoder` activation dtype correct.
4. Drop the training-only `action_model.beta_dist` sampler.

The returned object is the same policy instance, so `select_action` and
`predict_action_chunk` keep working.

## Why

The untouched policy already fails 2/10 development episodes, so the gate has no
margin. The dead modules and the compact embedding are exact. The remaining risk is
the decoder quantization. NF4 at 4 bits is the gentlest way to halve the INT8 decoder
storage. The sibling's naive NF4 decoder lost task 2 (7/10); this implementation
converts every Linear uniformly, keeps bf16 compute, and uses double-quantized
absmax, and it does **not** lose task 2.

## Last check output

`cd /tmp/work && /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 81.4% (resident 1026950986 of baseline 5532276502 bytes)
action error (max normalized difference): 0.078125
latency ms: median 140.73, p95 147.13
```

The action error is lower than the 0.683 sibling's 0.0947 despite the more aggressive
decoder quantization.

## Smoke rollout

Two independent `--phase smoke` runs (10 episodes, one per task, development seeds):

```
run 1: 8/10 successes; failures task 0 and task 8
run 2: 8/10 successes; failures task 0 and task 8
```

The untouched baseline fails the same two tasks, so this candidate loses no baseline
episode. The gate is met with the same 8/10 as the 0.683 sibling, at 81.4% memory
removed instead of 68.3%.

## What I would try next

- The 4-bit ceiling is close: NF4 packed weights compress only ~2.6% with zlib/bz2,
  so lossless post-compression is not worthwhile.
- Further gains need 3-bit/2-bit HQQ or AWQ-style activation-scaled INT4 on the
  decoder; that risks the no-margin gate and needs another smoke run.
- If a future gate has margin, quantize the remaining bf16 conv/position embeddings
  and the small non-64-aligned linears too (only ~15 MB total).
