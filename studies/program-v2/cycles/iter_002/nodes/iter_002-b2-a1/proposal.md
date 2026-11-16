# iter_002-b2-a1: NF4-packed inference core

## Change relative to parent (`iter_002-b2-a0`)
- Keep the parent's exact removals: delete the V-JEPA2 video encoder/predictor/processor, replace
  the unused tied `lm_head` with `nn.Identity`, and replace the 151936-row token embedding with a
  compact embedding over the 53 calibration token ids.
- Replace the parent's selective INT8 packing with **NF4** (4-bit normal-float, group size 32,
  FP16 group scales) on every remaining inference Linear:
  - all Qwen vision-tower Linears;
  - all text-decoder self-attention projections;
  - all text-decoder MLP projections (all 28 layers, not only the middle 3-26);
  - all action-head Linears (any Linear whose input width is odd falls back to INT8, since NF4
    nibble packing needs an even width).
- Each packed layer is dequantized inside its own `forward`, so no BF16 copy survives `compress`.
- All changes are in place on the evaluator's policy object; no compressed copy is left behind.

## Why
NF4 stores 4 bits/weight plus a small FP16 scale per 32 weights (~0.5625 bytes/weight), versus
~1.0625 bytes/weight for INT8. After the parent's exact removals, the remaining weights are the
vision tower, text attention/MLP and action head. Packing all of them to NF4 cuts the resident
weight set from ~2.68 GB to ~1.12 GB. The action error stayed at 0.0547 when the action head
moved to NF4, and stayed at 0.109 when the first/last text MLPs moved to NF4; text-attention NF4
raised it from 0.055 to 0.109. The 10-episode smoke rollout kept the same two failures as the
untouched policy.

## Last check output
```
status: ok
memory removed: 72.0% (resident 1570129408 of baseline 5604584960 bytes)
action error (max normalized difference): 0.109375
latency ms: median 324.33, p95 357.14
```

## Smoke rollout (same 10 development episodes as the gate)
```
status: ok, successes: 8/10
size_reduction: 0.7198493028108186
resident_bytes: 1570128384, baseline_resident_bytes: 5604584960
resident_before_inference_bytes: 1118204934
peak_cuda_bytes: 1570128384
```
The two failures are tasks 0 and 8, the same tasks the untouched policy fails on this smoke set.
All ten tasks otherwise succeeded, so the gate's 8/10 threshold is met with the same failure set
as the parent and the untouched policy.

## Next
- The resident figure is now peak-CUDA bound (1.12 GB weights vs 1.57 GB peak). The largest
  remaining opportunity is activation/peak reduction, not weight bytes:
  - chunk the NF4 dequantization along output rows so the temporary index/weight tensor for a
    25M-weight MLP layer is not materialized in full;
  - dequantize directly to the input dtype (BF16) instead of float32.
- A small additional weight cut is possible with NF4 group size 64 (saves ~0.03 GB) or by
  quantizing the remaining norms, but both are minor compared with peak reduction.
- If more quality margin is needed, keep the action head BF16 or INT8; the current NF4 action
  head is the most aggressive choice and should be the first rollback if a new episode fails.
