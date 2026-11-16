# Node iter_000-b2-a2 — exact peak trims + INT3 vision/action (lower action error)

Parent: `iter_000-b2-a1` (score 0.812084, INT3 text + INT4 vision/action, uint8 zero
points). This node keeps the parent's exact trims and its INT3 text recipe, attacks the
figure that actually binds the score (the CUDA peak during the episodes), and replaces the
INT4 tinygemm vision/action path with the same AWQ-scaled INT3 affine recipe used for text.

## What changed relative to the parent

1. **No KV cache (exact).** The deployed path is a single full forward
   (`_qwen_last_decoder_hidden` calls the inner `Qwen3VLModel` once, never a decode loop),
   but the config leaves `use_cache=True`, so the text decoder allocates a per-layer KV
   cache that grows to about 27 MB and is charged to the peak. `compress` now sets
   `use_cache=False` on every Qwen config that has the flag. The captured last-layer hidden
   state is unchanged.
2. **Chunked INT3 dequantization (exact weights).** The parent unpacked a whole text layer
   to bf16 inside `_Int3Linear.forward` (the largest is 6144x2048 = 25.2 MB). The new
   forward unpacks and dequantizes 480 output rows at a time (at most ~5.9 MB), writes each
   chunk into a preallocated bf16 output, and never materializes the full layer. To make a
   contiguous output-row chunk a contiguous packed-row range, `_pack3` now stores 10
   consecutive logical rows per int32 instead of the HQQ lane-interleaved layout. The
   packed values, scales and zeros are unchanged, so the dequantized weights are the same.
3. **INT3 vision and action (approximate, measured).** The vision tower and action head
   now use `_Int3Linear` with the same group-128 affine quantizer and AWQ input scaling as
   the text decoder, instead of the INT4 tinygemm path. This removes about 77 MB of weights
   and, more importantly, the INT4 `scales_zeros` convention was less accurate here: the
   measured action error **falls** from 0.256836 (parent) to 0.179199 even though the
   weights are smaller.
4. The parent's exact trims are unchanged: JEPA world model dropped, compact 53-row token
   embedding, `lm_head` = `nn.Identity`.

All surgery is in place on the same policy object; no bf16 copy of a quantized weight is
kept, no file is opened or written inside `compress`, and no workspace file is over 1 MiB.

## Last check output (`python tools/check.py`)

```
status: ok
memory removed: 83.4% (resident 930135552 of baseline 5604584960 bytes)
action error (max normalized difference): 0.179199
latency ms: median 1000.62, p95 1180.06
```

The parent measured resident 1053194752 and action error 0.256836. This node is 123 MB
smaller and has a 30% lower action error. The resident figure is the CUDA peak, which is
the binding one.

## Real gate evidence (evaluator's own smoke phase, same development seeds/states)

```
python -m rsi.gpu run --slots-dir /tmp/gpu-slots --n 1 -- \
  python -m evaluation.compression.study solution --workspace /tmp/work \
  --phase smoke --out /tmp/out/smoke_int3all
{"status": "ok", "phase": "smoke"}
successes: 8 / 10,  size_reduction: 0.8340404267865715
per_task: 0:0/1, 1:1/1, 2:1/1, 3:1/1, 4:1/1, 5:1/1, 6:1/1, 7:1/1, 8:0/1, 9:1/1
```

The two failures (tasks 0 and 8) are exactly the tasks the untouched policy, the INT4
sibling and the parent fail. The evaluator measured resident before inference 846.0 MB,
after inference 846.1 MB, peak 930.1 MB; the peak is the binding figure and it is the
packed INT3 weights plus the normal forward activation working set. The score is
`1 - 930134528 / 5604584960 = 0.83404`.

## Why this is not hiding memory

Nothing is parked, deferred or reloaded. The KV cache is never allocated; the chunked
dequantization holds one 480-row slice at a time and frees it before the next; the packed
INT3 tensors remain the resident representation. The returned object is the same policy.

## Next

- If the smoke gate holds, the next safe win is packing the per-group INT3 zero points
  3-bit (about 11 MB -> 4.1 MB, exact) and double-quantizing the bf16 group scales (about
  22 MB -> 11 MB, small error).
- More aggressive 2-bit text was already shown to destroy the policy in the parent; do not
  repeat it.
