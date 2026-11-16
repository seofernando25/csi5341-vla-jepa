# Node iter_000-b2-a1 — exact trims + INT3/INT4, with exact uint8 zero points

Parent: `iter_000-b2-a0` (score 0.810119, INT3 text + INT4 vision/action). This node keeps
the parent's deployed graph and quantization recipe, and makes one exact storage change:
the per-group zero points of the packed INT3 text linears are stored as `uint8` instead of
`bfloat16`. Zero points are integers in `[0, 7]`, so the dequantized weights are bit-identical;
the metadata drops from 22 MB to 11 MB.

## What changed relative to the parent

1. **Exact trims (unchanged from parent):** drop the JEPA world model
   (`video_encoder`, `video_predictor`, `video_processor`); replace the 151936x2048 token
   embedding with the 53 prompt rows (`calibration["token_ids"]`) plus a lookup; replace the
   tied, never-called `lm_head` with `nn.Identity()`.
2. **Text decoder (unchanged recipe):** 196 linears stored as packed 3-bit affine groups
   (group 128, 10 values per int32) with AWQ input scaling.
3. **Vision tower and action head (unchanged recipe):** INT4 tinygemm groups with AWQ input
   scaling; tiny linears stay bf16.
4. **New, exact:** `_Int3Linear.zeros` is a `uint8` buffer and is cast to bf16 only inside
   `forward` for the dequantization. This is a pure dtype change of integer metadata, so the
   reconstructed weights and actions are unchanged.

All surgery is in place on the same policy object; old modules are dropped and
`gc.collect()` + `torch.cuda.empty_cache()` run before returning. No file is opened or
written inside `compress`, no bf16 copy of a quantized weight is kept, and no workspace file
is over 1 MiB.

## Rejected experiment (important negative result)

The parent's calibration says text blocks 3-26 have tiny residual influence, so I tested
structured MLP channel pruning (Wanda-style ranking of the 6144 MLP hidden channels, applied
before quantization) and full text-block skipping:

| variant | action error | resident | smoke |
|---|---|---|---|
| parent (INT3 text, INT4 vision/action) | 0.257 | 1064 MB | 8/10 |
| prune 25% text-MLP channels | 0.371 | 983 MB | **7/10** (task 4 newly fails) |
| prune 50% text-MLP channels | 0.487 | 815 MB | **5/10** (tasks 1, 4, 5 newly fail) |
| prune 50% + INT4 text | 0.469 | 906 MB | not run |
| skip text blocks 3-26 | 1.578 | — | not run |
| skip text blocks 3-10 | 0.967 | — | not run |

The block-influence signal is misleading because the residual stream is dominated by
massive activations; removing even 8 text blocks destroys the policy. The pruning code was
therefore disabled (`PRUNE_FRACTION = 0.0`), leaving the parent's proven quantization path.
The pruning helpers remain in `solution.py` but are inert.

## Last check output (`python tools/check.py`)

```
status: ok
memory removed: 81.2% (resident 1053194752 of baseline 5604584960 bytes)
action error (max normalized difference): 0.256836
latency ms: median 366.24, p95 455.76
```

The parent measured `resident 1064204800`, action error `0.256836`; the new resident is
`1053194752`, i.e. 11,010,048 bytes less with exactly the same action error. The binding
figure is the CUDA peak during the episodes, and the 11 MB metadata reduction lowers it
one-for-one. The parent's real smoke gate was 8/10 with the same failing tasks (0 and 8) as
the untouched policy; because this change is bit-exact in the dequantized weights, that
gate result carries over.

## Why this is not hiding memory

Nothing is parked, deferred or reloaded. The world model and dead embedding rows are
dropped and collected, `lm_head` is unregistered, and the packed INT3/INT4 tensors are the
resident representation. The only transient is one dequantized text layer inside its
`forward`, freed immediately after the matmul. No workspace file is over 1 MiB.

## Next

- Chunk the INT3 dequantization by output rows to cut the ~25 MB full-layer transient. The
  packed layout is group-major (10 q-rows per int32), so a naive contiguous slice is wrong;
  it needs a group-aware unpack. A correct version should save ~20-30 MB of peak.
- Double-quantize the remaining bf16 group scales (about 22 MB) with a per-row fp16 scale;
  this is the next exact-ish metadata win.
- If more risk is acceptable, INT3 for the vision tower (about 47 MB) or action head (about
  15 MB) needs a real smoke run: the 0.25-pruning result shows the 8/10 gate has no margin.
