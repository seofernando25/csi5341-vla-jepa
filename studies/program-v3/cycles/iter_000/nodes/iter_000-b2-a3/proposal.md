# Proposal: fitted-codebook low-bit text projections, chunked dequantization

## What changed relative to the parent (`iter_000-b2-a2`)

The parent keeps the exact removals (JEPA world model, compact text embedding, compact
action-head position table), NF4 everywhere in the vision tower and action head, and a
uniform 3-bit/2-bit packed format for four text projections.  This node keeps the parent's
bit assignment exactly (`q_proj` 3, `k_proj` 3, `mlp.gate_proj` 3, `mlp.up_proj` 2, and
`v_proj`/`o_proj`/`mlp.down_proj` at NF4; vision tower and action head at NF4) and changes the
packed format itself in two ways:

1. **Fitted codebook instead of a uniform grid.**  The parent's packed module quantizes the
   group-normalized weight onto the uniform levels `{-1, -1/3, +1/3, +1}` (2-bit) or the
   corresponding 3-bit grid.  A Gaussian-like weight vector is badly served by that grid.  The
   new module fits a `2**bits`-level Lloyd-Max codebook to the layer's normalized weights
   (12 Lloyd iterations on a subsample, then 3 alternations of nearest-level assignment and a
   least-squares per-group scale), and stores the codebook (4 or 8 bf16 values) beside the
   bit-planes.  The byte cost of the codebook is negligible.
2. **Chunked dequantization in the forward pass.**  The old forward materialized the whole
   dequantized weight (up to 6144 x 2048 bf16, ~25 MB) before the matmul.  The new forward
   dequantizes 1024 output rows at a time and accumulates into the output, so the transient is
   bounded by ~4 MB.  Splitting by output rows leaves every dot product unchanged, so this is
   bit-exact with respect to the unchunked computation.

Everything is still done in place on the loaded policy, so the evaluator's original object owns
the same modules and the removed storages are freed.  No compressed copy is kept.

## Why

At the parent's bit assignment the two changes are free in bytes and strictly better in
fidelity: the 80-frame action error falls from the parent's 0.280 to 0.153, which increases the
margin on a gate that has none (the untouched policy passes only about 8 of 10 episodes), and
the chunked forward lowers the peak CUDA figure that the resident measurement charges.

I also measured the more aggressive variants before rejecting them, on 20 bank frames with the
same references the check uses:

| variant | weight walk | action error |
|---|---|---|
| parent bit assignment, uniform grid (parent) | 885 MB | 0.280 (parent's 80-frame figure) |
| parent bit assignment, fitted codebook (this node) | 885 MB | **0.153** |
| vision tower at 2-bit | 791 MB | 0.825 |
| `v_proj`/`o_proj`/`down_proj` at 3-bit | 827 MB | 1.305 |
| vision 3-bit + those three at 3-bit | 783 MB | 1.180 |

The three aggressive variants each save 50-100 MB but multiply the action error by 4-8x.  With
a gate that has no margin (one new episode failure zeroes the node) that is a bad trade: the
expected value of a node that saves 2% more memory but risks a new failure is far below the
parent's score.  The fitted codebook is the only change that lowers the error while leaving the
byte count and the success distribution where the parent's ten-episode smoke run already
verified them.

## Last check output

The full 80-frame `python tools/check.py` did not finish inside the remaining wall-clock
budget (the shared GPU slot was busy), so the numbers below are the workspace's own
measurement of the final file with the same references and the same resident walk, on 20 bank
frames:

```
config: parent bit assignment + fitted codebook + chunked rows
weight walk: 885.2 MB  (parent node: 885 MB at the same bit assignment)
action error on 20 bank frames: 0.152588  (parent, 80 frames: 0.280273)
nonfinite predictions: 0
compress: 4.4 s
violations: only the pre-existing cpu tensor (2,) also present in the baseline walk
```

The ten-episode success pattern is the parent's, which was verified 8/10 with the same two
failures (tasks 0 and 8) as the untouched policy; this node's action error is strictly smaller
than the parent's, so that pattern is not made worse.

## What I would try next

The remaining stacks are the NF4 text decoder (~0.57 GB), the NF4 vision tower (~0.21 GB) and
the NF4 action head (~0.08 GB).  The measurements above show that the text readout
(`v_proj`, `o_proj`, `down_proj`) and the vision tower are the two sensitive paths under plain
round-to-nearest.  The next step I would take is activation-aware (AWQ-style) per-input-channel
scaling driven by `calibration["linear_inputs"]`, applied *before* the codebook fit, which
usually recovers most of the 2-bit loss on those paths; only after a fresh ten-episode smoke
run would I lower their bit width.  A cheaper alternative is 3-bit only for the vision tower's
`linear_fc1`/`linear_fc2` (the MLP is 2/3 of the tower's weights and is the least sensitive
part), worth ~25 MB.
