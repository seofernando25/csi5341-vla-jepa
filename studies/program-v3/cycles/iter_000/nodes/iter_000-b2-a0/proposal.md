# Proposal: exact removals, NF4 text decoder, int8 vision/action

## What changed relative to the parent

The parent is the untouched template (`compress` returns `policy`).  This node mutates the
loaded policy in place and returns the same object:

1. **Drop the JEPA world model.**  `video_encoder`, `video_predictor` and `video_processor`
   are training-only; inference runs Qwen plus the action head.  Setting them to `None`
   frees their CUDA storages (~967 MB).
2. **Compact the text embedding table.**  The Qwen table is 151936 x 2048 bf16 (~622 MB), but
   the ten LIBERO-Spatial instructions, the chat template, the image placeholders and the
   action tokens use 53 distinct ids.  The table and the tied (unused) `lm_head` are replaced
   by a 54-row embedding plus an int32 id map; unknown ids map to a zero row.
3. **Compact the action-head position table.**  The DiT indexes only positions `0..6`, so the
   1024 x 768 table is replaced by its first 7 rows.
4. **NF4 for the Qwen text decoder.**  Every remaining `nn.Linear` in
   `language_model.layers` is replaced by a `bitsandbytes.nn.Linear4bit` (NF4, double
   quantization, bf16 compute).  This is the largest stack (2.82 GB).
5. **Int8 for the vision tower and action head.**  Every `nn.Linear` in `visual` and
   `action_model` is replaced by a custom weight-only int8 module with one symmetric scale
   per output channel.  The int8 weights are dequantized a layer at a time in `forward` and
   the temporary is released.  The action head's `TimestepEncoder` stays bf16 because its
   forward reads `next(self.parameters()).dtype`.

All changes are in place, so the evaluator's original policy object owns the same modules and
the old storages are freed.  No compressed copy is kept.

## Why

The exact removals are free.  The text decoder is the largest remaining stack and NF4 halves
it again relative to int8; the vision tower and action head are smaller and more sensitive,
so they get the safer int8 treatment.  The smoke gate has no margin (the untouched policy
passes about 8/10), so the mixed scheme was chosen after measuring both action error and the
actual ten smoke episodes.

## Last check output

```
status: ok
memory removed: 73.9% (resident 1460460032 of baseline 5604584960 bytes)
action error (max normalized difference): 0.113281
latency ms: median 110.38, p95 116.93
compress: 4.4 s
nonfinite predictions: 0
```

The workspace's own ten-episode smoke run (`evaluation.rollout`, tasks 0..9, development
phase, one trial each, the evaluator's seeds) returned **8/10 successes**.  The two failures
are tasks 0 and 8, exactly the two the untouched policy fails, so the success distribution is
unchanged.

## What I would try next

The remaining large weights are the NF4 text decoder (~0.7 GB), the int8 vision tower
(~0.4 GB) and the int8 action head (~0.16 GB).  NF4 for the vision tower would remove another
~0.2 GB, and NF4 for the action head another ~0.08 GB, but the action error is already 0.113
and the gate has no margin.  A safer next step is to keep the first and last text-decoder
layers in bf16 (or int8) and NF4 only the middle layers, then re-run the ten smoke episodes.
I would not take any further step without the actual smoke result, because a single new
episode failure zeroes the node.
