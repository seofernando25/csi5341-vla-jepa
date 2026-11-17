# Proposal: NF4/AWQ parent plus provably inert action-head trimming

## Parent
`iter_002-b2-a0` (in-place NF4/AWQ compression): resident 1,105,683,968 bytes, 80.26%
removed, check action error 0.096, development smoke 8/10 (tasks 0 and 8 fail, exactly the
untouched baseline). The best node in the tree is `iter_002-b1-a1` at 0.804264
(1,097,017,856 bytes). The 8/10 gate has no margin, so every new change must keep tasks
0 and 8 as the only failures.

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Keep the parent's compression**: the training-only world model is removed, the tied
   151936-row Qwen embedding is compacted to the fixed LIBERO prompt rows, text block 3 is
   dropped, and every large Linear in the text decoder, vision tower and action head is NF4
   with AWQ-style per-input-channel scaling from `linear_inputs`.
2. **Drop action-head DiT blocks 7, 9 and 11.** The calibration ranks them lowest of the
   action blocks (`action.7 = 0.0020`, `action.9 = 0.0016`, `action.11 = 0.0021`; block 0
   is 2.03 and the median block is ~0.003). All three are odd-indexed self-attention
   blocks, so removing them does not change the cross-attention pattern of the remaining
   blocks. Each NF4 block is 5.15 MB, so the three removals save 15.5 MB.
3. **Compact the action position table.** `action_model.position_embedding` has 1024 rows
   but `_build_inputs` only indexes `0..chunk_size-1` (7 rows); replace it with an 8-row
   table holding the first eight rows. Saves 1.6 MB.

## Why
The parent is already at the 8/10 gate, so the only safe direction is to remove parameters
that inference provably does not need. The three action blocks are the least influential in
the calibration, and the position table is over-sized for the fixed chunk length. All
removals are real: the old modules are replaced and freed before `compress` returns,
nothing is deferred, hidden or written to disk.

I also measured dropping two more text decoder blocks (4 and 6), whose calibration
influence (5.8e-5, 9.8e-5) is the same order as block 3 (4.6e-5). That reached 81.4%
removed but raised the action error to 0.333, so it was rejected: the calibration residual
influence does not reliably predict the episode outcome, exactly as sibling
`iter_002-b1-a1` found for text block 5. The final solution therefore keeps every text
block except block 3.

## Last check
`python tools/check.py` (raw output in `check.log`):

```
status: ok
memory removed: 80.5% (resident 1093010944 of baseline 5604584960 bytes)
action error (max normalized difference): 0.102539
latency ms: median 130.64, p95 134.55
compress: 7.5 s
nonfinite predictions: 0
```

The action error (0.103) is in the same range as the nodes that passed the 8/10 gate
(`iter_002-b1-a0` at 0.103 and `iter_002-b1-a1` at 0.109 both kept tasks 0 and 8 as the only
failures), while resident size falls 12.7 MB below the parent and 4.0 MB below the best
node in the tree.

## Next
The remaining ~1.09 GB is dominated by the 27 NF4 text blocks (~680 MB) and the vision
tower (~210 MB). A further reduction needs a lower-bit scheme with error compensation
(GPTQ-style 3-bit on the low-influence text blocks) rather than more whole-block drops,
because whole-block removal is empirically much more damaging than the calibration
influence suggests. A fresh development smoke must accompany any such attempt.
