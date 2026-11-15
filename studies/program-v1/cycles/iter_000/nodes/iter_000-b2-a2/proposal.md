# Node iter_000-b2-a2

## Change relative to parent (iter_000-b2-a1)

The parent deployed packed int3 on the text decoder and vision tower, packed int4
(tinygemm) on the action head, group size 128, bf16 scales and uint8 zero points.  It
scored 0.849365 with 8/10 episodes.  This node keeps the same inference-exact structural
changes (world-model removal, compact 53-row token embedding, unused LM head) and the
same text/vision int3 recipe, and changes only the quantizer bit width and metadata:

1. **Action head: packed int3 instead of int4.**  The DiT action head is 155 M
   parameters and was the only block still at 4 bits.  It now uses the same packed
   affine int3 path as text/vision (AWQ input scale, 8 levels, group 128).  This is
   exactly the recipe node iter_000-b0-a2 smoke-tested at 8/10, so it is not new risk.
2. **Zero points packed two per byte.**  For 2/3-bit weights the zero points are
   integers in 0..7, so two of them fit in one uint8.  This halves the zero metadata and
   is bit-exact: the dequantized weights are identical.
3. **fp8 scales with a per-module multiplier.**  Each module's group scales are divided
   by one bf16 multiplier and stored as `float8_e4m3fn` (1 byte per group instead of 2).
   The multiplier keeps every scale in the normal fp8 range.  The action error moves
   from 0.207 (bf16 scales) to 0.217, so this costs a little quality but stays on the
   right side of the gate.

Together (2) and (3) remove ~23 MB of quantizer metadata without changing the number of
quantized weights.  All replacements happen in place, so the original storages are freed;
no compressed copy is kept.  `compress` performs no file I/O and writes no files.

## Last check output

`python tools/check.py` on this workspace:

```
status: ok
memory removed: 86.0% (resident 773712674 of baseline 5532276502 bytes)
action error (max normalized difference): 0.217285
latency ms: median 826.38, p95 878.20
```

The latency is inflated by the shared GPU; it is not a scoring input.

## Smoke rollout (the evaluator's 10-episode phase)

```
status: ok, successes 8/10
per task: 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:1
resident 773,712,674 bytes, size_reduction 0.8601456970
```

Tasks 0 and 8 are the two episodes the untouched policy also fails; every task the
parent passed still passes.  This beats the tree's previous best (0.855983).

## Experiments tried and rejected

- **Group size 256 for text+vision (metadata halved), with the same action-int3 and
  fp8-scale changes:** 86.13% removed (767,237,154 bytes) but action error rose to
  0.242 and the smoke rollout flipped task 4 (7/10).  Reverted to group 128; the
  metadata bytes were recovered losslessly with packed zeros instead.

## What I would try next

- Action head at int2 (another ~19 MB, ~0.35 points) with the same AWQ recipe; the
  action head barely moved between int4 (0.198) and int3 (0.207), so int2 is the next
  cheap candidate, but it needs a smoke rollout.
- Text `down_proj` at int2 on the lowest-influence half of the layers: node b0-a2 showed
  int2 on *all* down-projections flips a task, so a half-dose is worth ~22 MB.
- A per-group 4-centroid codebook for int2 (instead of the affine 4-level grid) is the
  main route to a real text-int2 win; the affine int2 error is just over the quality edge.
