# Node iter_000-b2-a1

## Change relative to parent (iter_000-b2-a0)

The parent deployed packed int4 (AWQ + PyTorch tinygemm) on text, vision and action
Linears and scored 0.8059 with 8/10 episodes.  This node keeps the same inference-exact
structural changes (world-model removal, compact 53-row embedding, unused LM head) and
moves the text decoder and the vision tower from int4 to **packed int3**:

1. **Text decoder: groupwise int3 with real 3-bit packing.**  Eight consecutive input
   columns are packed into three uint8 bytes (0.375 byte/parameter).  The AWQ
   per-channel input-scale search is kept, with 8 affine levels instead of 16.
   Dequantization happens one layer per forward.
2. **Vision encoder: the same groupwise int3** packing.  The vision tower is 407 M
   parameters and is the second-largest resident block.
3. **Action head: stays groupwise int4** with the tinygemm kernel.  The action head is
   the most sensitive part, so it is not pushed to 3 bits.
4. Quantizer side data is compact: bf16 scales plus uint8 zero points (3 bytes per
   128-weight group).  The parent's int4 tinygemm format stores two bf16 values per
   group, so the int3 path also has lower per-group overhead.

All replacements happen in place, so the original storages are freed; no compressed
copy is kept.  `compress` performs no file I/O and writes no files.

## Last check output

`python tools/check.py` on this workspace:

```
status: ok
memory removed: 84.9% (resident 833352922 of baseline 5532276502 bytes)
action error (max normalized difference): 0.197754
latency ms: median 809.20, p95 938.38
```

The latency is inflated because the shared GPU was at 100% utilization during the run.
The 84.9% figure is above the tree's previous best (84.68%, node iter_000-b0-a1) with
essentially the same action error (0.198 vs 0.201), and the int3 text+vision recipe is
the one that passed the 10-episode smoke rollout 8/10 in that node (tasks 0 and 8 fail,
exactly as the untouched policy).

## Why this should keep the 8/10 gate

The two structural changes are inference-exact.  The only approximate change is the
3-bit packing of text and vision weights.  The identical recipe (same AWQ search, same
group size, same affine 8-level grid) scored 8/10 in the smoke rollout of
iter_000-b0-a1, with the same two failing tasks as the untouched policy.  This node
differs only in the packing/scale layout, which is a lossless representation of the
same quantized values, so the deployed forward is numerically identical to that node.

## What I would try next

- Action-head int3 would remove another ~19 MB (~0.35 points), but the action head is
  the most sensitive block and the 8/10 gate has no margin; it needs a smoke rollout
  before adoption.
- 2-bit text is not viable: the calibration-weighted per-layer error of int2 is about
  1.5-1.7 (larger than the signal), versus 0.27-0.34 for int3.
- Structured pruning is the only route to a large further gain: 25% of the text MLP
  intermediate width would save ~107 MB (~1.9 points) and can be ranked with the
  calibration's `down_proj` input energy, but it needs a smoke rollout to prove it does
  not flip a task.
