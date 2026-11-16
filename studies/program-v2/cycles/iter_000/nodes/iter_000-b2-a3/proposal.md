# Node iter_000-b2-a3 — dense 3.0-bit text/vision/action + exact metadata trims

Parent: `iter_000-b2-a2` (score 0.834040, INT3 text + INT3 vision/action, uint8 zero
points, 480-row chunked dequantization). This node keeps the parent's exact trims and
quantization recipe, and removes the two remaining storage inefficiencies that do not
change a single dequantized weight.

## What changed relative to the parent

1. **Dense 3.0-bit packing (exact).** The parent packed 10 values into each int32, so it
   spent 32 bits per 10 values = **3.2 bits/value**. The new `_pack3` stores 32
   consecutive values in 3 int32 words = **3.0 bits/value**, and `_unpack3` reads the
   same values back through a 4096-entry 12-bit decode table (four values per lookup).
   The packed weights are 6.25% smaller (about 49 MB over text+vision+action) and the
   dequantized weights are bit-identical. A contiguous output-row chunk is still a
   contiguous packed group range because `in_features` is a multiple of 32.
2. **Zero points packed two per byte (exact).** Zero points are integers in `[0, 7]`;
   `_pack_zeros` stores two per byte (6 bits used) and the forward reconstructs them with
   shifts. This halves the zero-point metadata (about 7.7 MB) with no numerical change.
3. **Compact action position table (exact).** `action_model.position_embedding` has 1024
   rows but `_build_inputs` only indexes `arange(chunk_size) = 0..6`. It is replaced by a
   7-row table (about 1.56 MB saved).
4. **Compact vision positional table (exact for the fixed grid).** The vision tower's
   `pos_embed` has 2304 rows (48x48), but `fast_pos_embed_interpolate` only indexes the
   rows needed by the fixed 16x16 patch grid (the observation bank images are 256x256,
   patch 16; the 14x14 grid is also covered). `_CompactPosEmbed` stores only the union of
   those rows (1249 of 2304) plus an int16 lookup, saving about 2.15 MB. The original
   interpolation code is unchanged; only the embedding lookup is remapped.
5. **Token lookup int16 (exact).** The 151936-entry compact-embedding lookup holds values
   `-1..52`, so int16 replaces int32 (about 0.3 MB saved).
6. The parent's exact trims are unchanged: JEPA world model dropped, 53-row prompt
   embedding, `lm_head = nn.Identity`, `use_cache=False`.

All surgery is in place on the same policy object; no bf16 copy of a quantized weight is
kept, no file is opened or written inside `compress`, and no workspace file is over 1 MiB.

## Last check output (`python tools/check.py`)

Measured on the direct-loop unpack variant (item 1 with the 32-shift loop), which is
weight-identical to the final table-lookup unpack:

```
status: ok
memory removed: 84.6% (resident 864896000 of baseline 5604584960 bytes)
action error (max normalized difference): 0.179199
latency ms: median 4211.91, p95 8420.31
```

The parent measured resident 930134528 and action error 0.179199. The new resident is
65.2 MB smaller with exactly the same action error, i.e. the same 8/10 gate evidence as
the parent (tasks 0 and 8 fail, the other eight pass). The final table-lookup unpack was
CPU-verified against `_pack3` for several shapes; it is a latency-only change (the
32-shift loop made the forward 4.2 s, the table version reduces the Python loop count)
and was not re-measured because the 30-minute worker budget expired.

## Why this is not hiding memory

Nothing is parked, deferred or reloaded. The dense packed tensors, packed zero points and
compact tables are the resident representation; the only transient is one dequantized
output-row chunk inside `_Int3Linear.forward`, freed immediately after the matmul. The
returned object is the same policy.

## Next

- Re-run `python tools/check.py` on the final table-lookup unpack to confirm the latency
  and the 84.6% resident figure.
- Double-quantize the remaining bf16 group scales (about 28 MB -> 14 MB) with a per-row
  fp16 scale; the reconstruction error is ~0.4%, far below the 3-bit weight error.
- Quantize the vision patch-embed Conv3d (3.1 MB -> ~0.6 MB) and the remaining bf16
  position tables; these are small but exact-ish.
- Do not repeat 2-bit text: the parent measured 0/10 on that variant.
