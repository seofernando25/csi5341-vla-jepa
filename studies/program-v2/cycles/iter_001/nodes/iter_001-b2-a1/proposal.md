# iter_001-b2-a1 — INT4 weight-only compression of the live Qwen/action backbone

## What changed relative to the parent (`iter_001-b2-a0`)

The parent already removed the inference-dead payloads exactly:

1. the V-JEPA world model (`video_encoder`, `video_predictor`, `video_processor`);
2. the tied `lm_head`, which `_qwen_last_decoder_hidden` never calls;
3. all but the 53 prompt-embedding rows seen in the frozen calibration.

That leaves ~4.0 GB of genuinely live weights: the Qwen text decoder, the Qwen
vision tower, and the DiT action head.  This node compresses those in place:

- **INT4 weight-only quantization of every large Linear layer** in the text,
  vision and action roots.  Weights are packed with PyTorch's Ampere-compatible
  tinygemm layout (`torch._convert_weight_to_int4pack` /
  `torch._weight_int4pack_mm`), with affine group size 128 (scales and zeros
  stored in bf16).  Activations and biases stay bf16.  The original bf16
  weights are replaced and released, not copied.
- **Action-head position embedding compaction**: the head indexes only
  positions `0..action_horizon-1` (7 rows), so the 1024-row table is replaced by
  its first 7 rows.  This is exact.
- **Small embedding lookup cleanup**: the compact embedding's lookup table is
  int32 instead of int64, and the padding row is retained.

The quantization is deliberately weight-only: no activation quantization, no
layer/head removal, and no change to the inference graph.  The measured action
error on the 80 held-out frames is 0.140625 maximum normalized difference, and
the 10-episode smoke run keeps exactly the parent's success pattern (tasks 0 and
8 fail, the other eight succeed).

## Last check output

`python tools/check.py` (final solution.py):

```
status: ok
memory removed: 79.6% (resident 1142731264 of baseline 5604584960 bytes)
action error (max normalized difference): 0.140625
latency ms: median 165.61, p95 259.28
```

The evaluator's 10-episode smoke phase on the same workspace:

```
status: ok, phase: smoke
successes: 8 / 10
size_reduction: 0.7961079637197613
resident_bytes: 1142730240, baseline_resident_bytes: 5604584960
peak_cuda_bytes: 1142730240
per-task: task 0 fail, tasks 1-7 pass, task 8 fail, task 9 pass
```

This is the same per-task pattern as the untouched policy and as the parent,
so the smoke gate remains at 8/10 with the same two pre-existing failures.

## What I would try next

The remaining resident bytes are almost entirely the 4-bit packed weights plus
the ~72 MB inference activation peak; exact structural removal is exhausted.
The next step would be a mixed-precision split: keep attention and the action
head at INT4/INT8, and try 3-bit or 2-bit HQQ only on the text/vision MLP
matrices, accepting the larger action error only if a smoke run still shows
8/10.  A safer intermediate is group size 64 for the INT4 scales (slightly more
memory, lower action error), followed by another smoke run.  I would not remove
whole decoder blocks: the calibration block influences are small for many text
layers, but the gate has no margin, so a lossy structural change is riskier than
the measured 4-bit weight-only change.
