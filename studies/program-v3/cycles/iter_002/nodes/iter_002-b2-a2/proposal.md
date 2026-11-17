# Proposal: packed 3-bit text MLP + action DiT, and no KV cache

## Parent
`iter_002-b2-a1` (in-place NF4/AWQ compression): resident 1,093,016,576 bytes, 80.50%
removed, check action error 0.102539, development smoke 8/10 (tasks 0 and 8 fail, exactly
the untouched baseline). The 8/10 gate has no margin, so every new change must keep tasks
0 and 8 as the only failures.

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Keep the parent's compression**: the training-only world model is removed, the tied
   151936-row Qwen embedding is compacted to the fixed LIBERO prompt rows, text block 3 is
   dropped, action DiT blocks 7, 9 and 11 are dropped, the action position table is reduced
   to the 8 rows `_build_inputs` can index, and every large Linear in the text decoder,
   vision tower and action head is quantized.
2. **Packed 3-bit text MLP.** Every `gate_proj`, `up_proj` and `down_proj` in the 27 text
   decoder blocks is stored as 3-bit codes (8 codes per 3 bytes) with one bf16 scale per
   group of 64 input channels. The 8-level codebook is the normal-quantile map normalized
   to [-1, 1] and keeps an exact zero. The AWQ per-input-channel scale from
   `linear_inputs` is folded into the stored weight and divided out in forward. The weight
   is dequantized in chunks of 256 output rows, which bounds the transient bf16 weight.
   This replaces the parent's NF4 on the text MLP and saves about 111 MB.
3. **Packed 3-bit action DiT.** The action DiT's attention and feed-forward projections
   (shapes 768x768, 3072x768, 768x3072) use the same packed 3-bit scheme with full
   dequantization (the largest is only 4.7 MB bf16). This saves about 9 MB.
4. **Disable the Qwen KV cache.** `predict_action_chunk` is a single forward pass, so the
   per-layer key/value cache is pure overhead. Setting `use_cache=False` on the Qwen text
   config is bit-exact and removes about 21 MB from the measured peak.

Everything else (text/vision attention, vision MLP, action decoder, tiny projections)
stays NF4/AWQ or bfloat16 as in the parent.

## Why
The parent is already at the 8/10 gate, so the only useful direction is lower-bit storage
of weights whose quantization error stays inside the parent's error band. The text MLP is
75% of the text decoder parameters and tolerated 3-bit quantization well: the fixed
normal-quantile codebook gave action error 0.125 on 40 bank frames, versus 0.191 for a
per-weight k-means codebook (the k-means centers lost the exact zero). The action DiT
3-bit alone measured 0.102, the same as the parent's NF4 action head, and the combination
measured 0.116 on 40 bank frames. Disabling the KV cache is exact.

Measured and rejected during this search:

- 3-bit text attention (q/k/v/o): action error 0.215; k/v only 0.167; o_proj only 0.189.
- 3-bit vision MLP: 0.146 alone, 0.170 combined with the text MLP; 3-bit vision attention
  was 0.215. Not worth the extra ~22 MB for the error.
- 2-bit text MLP: 4.90 action error (far too coarse).
- Dropping the vision deepstack mergers: 0.876 action error (deepstack matters a lot).
- Dropping vision blocks 12-17: 0.446; dropping text block 4: 0.268; dropping action
  blocks 5 and 6 in addition to 7/9/11: 0.150 (only 8 MB, not worth the risk).

## Last check
`python tools/check.py` (final run, raw output in `check.log`):

```
status: ok
memory removed: 82.8% (resident 965061632 of baseline 5604584960 bytes)
action error (max normalized difference): 0.144043
latency ms: median 503.08, p95 599.13
compress: 4.9 s
nonfinite predictions: 0
```

Development smoke with the repository's own LIBERO rollout code, one episode per task:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1  task=4 success=1
task=5 success=1  task=6 success=1  task=7 success=1  task=8 success=0  task=9 success=1
successes=8/10
```

Tasks 0 and 8 are the same two failures as the untouched policy and the parent, so the
no-margin 8/10 gate is preserved while resident memory falls from 1,093,016,576 to
965,061,632 bytes (80.50% -> 82.8% removed).

## Next
The remaining ~965 MB is mostly the 27 NF4 text attention projections (~180 MB), the NF4
vision tower (~210 MB), and the 3-bit text MLP (~400 MB). A further safe reduction would
need either a smaller group size only on the low-influence text layers, GPTQ-style error
compensation for 3-bit attention, or a small (5%) activation-aware pruning of the text MLP
intermediate channels. Any of those needs a fresh development smoke because the gate has
no margin; whole-block removal is empirically much more damaging than the calibration
influence suggests.
