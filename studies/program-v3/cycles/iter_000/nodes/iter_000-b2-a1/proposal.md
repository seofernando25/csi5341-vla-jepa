# Proposal: NF4 for the vision tower and action head, on top of the parent's exact removals

## What changed relative to the parent (`iter_000-b2-a0`)

The parent already does the exact removals (world model, compact text embedding, compact
action position table) and quantizes the text decoder to NF4 while the vision tower and action
head use weight-only int8.  This node keeps all of that and changes one thing:

**Every remaining `nn.Linear` leaf in the vision tower and the action head is now a
bitsandbytes NF4 module** (double quantization, bf16 compute), replacing the parent's int8
modules.  The action head's `TimestepEncoder` stays bf16 because its forward reads
`next(self.parameters()).dtype`; everything else (vision blocks, mergers, DiT attention,
feed-forward, AdaLayerNorm, action encoder/decoder) is NF4.

All changes are in place, so the evaluator's original policy object owns the same modules and
the old storages are freed.  No compressed copy is kept.

## Why

The parent's resident figure is `max(weight walk, peak CUDA)`; the peak is only ~164 MB above
the weights, so every weight byte removed lowers the score directly.  After the parent's
removals the remaining stacks were the NF4 text decoder (~0.70 GB), the int8 vision tower
(~0.40 GB) and the int8 action head (~0.16 GB).  NF4 halves the two int8 stacks again
(~0.28 GB saved) with the same bf16 compute path the text decoder already uses.  The measured
80-frame action error actually improved from the parent's 0.113 to 0.086, and the real
ten-episode smoke run kept the untouched policy's success pattern.

I also tried the calibration's `block_influence` as a pruning guide: dropping text blocks
3..10 (influence 4.6e-5..2.2e-4, far below blocks 0..2 and 27) reached 83.7% removed but the
80-frame action error jumped to 1.066.  The influence metric is relative to the huge residual
norm of block 2, so it does not predict the action readout's sensitivity.  That variant was
rejected and is not in `solution.py`.

## Last check output

```
status: ok
memory removed: 79.8% (resident 1131857920 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0859375
latency ms: median 142.72, p95 159.69
compress: 6.2 s
nonfinite predictions: 0
```

The workspace's own ten-episode smoke run (`evaluation.rollout.run_episode`, tasks 0..9,
development phase, one trial each, the evaluator's seeds) returned **8/10 successes**:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1
task=4 success=1  task=5 success=1  task=6 success=1  task=7 success=1
task=8 success=0  task=9 success=1
TOTAL 8 / 10 [0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
```

The two failures are tasks 0 and 8, exactly the two the untouched policy and the parent fail,
so the success distribution is unchanged while the resident memory fell from the parent's
1.460 GB to 1.132 GB.

## What I would try next

The remaining weights are the NF4 text decoder (~0.70 GB), NF4 vision tower (~0.20 GB) and
NF4 action head (~0.08 GB).  A packed 3-bit or 2-bit weight format for the text decoder would
remove another 0.18-0.35 GB, but it must be checked against the actual ten-episode smoke run,
not just the 80-frame action error: the block-drop experiment shows how poorly a low action
error predicts the readout's sensitivity.  A safer next step is 3-bit only for the vision
tower's later blocks (12..22 have the lowest vision influence), which is worth ~0.05 GB and
does not touch the text readout.  I would not take either step without a fresh smoke run,
because one new episode failure zeroes the node.
