# iter_002-b2-a2: chunked NF4 dequantization + group 64 (peak-CUDA cut)

## Change relative to parent (`iter_002-b2-a1`)
Keep the parent's exact removals (V-JEPA2 encoder/predictor/processor, tied `lm_head`, compact
token embedding over the calibration token ids) and its NF4 packing of every remaining inference
Linear (text attention, text MLP, vision tower, action head; odd input widths fall back to INT8).
Two changes:

1. **Chunked dequantization inside each packed layer.** The parent materialized the whole
   dequantized weight with int64 index tensors before the GEMM. For a 12.6M-weight text-MLP layer
   that is ~300 MB of int64 indices plus ~100 MB of float32 copies alive at once, which set the
   evaluator's peak-CUDA figure (1.57 GB peak against 1.12 GB of packed weights). The new forward
   allocates the final weight buffer once and fills it in chunks of ~1.5M weights; each chunk
   decodes packed bytes with a single 256-row embedding lookup
   (`byte_table[byte] = [codebook[low nibble], codebook[high nibble]]`) instead of shift + int64
   scatter, scales in float32, and copies into the buffer. The single GEMM then runs on the same
   values in the same order as the parent, so outputs are **bit-identical** (verified for NF4 and
   INT8, odd/even widths, and under autocast with fp32 inputs).
2. **NF4 group size 32 → 64** (still FP16 group scales). This halves the scale metadata
   (~62 MB across the model) at the standard bitsandbytes block size. Measured action error on the
   80 held-out frames improved from 0.109375 to 0.0820312, and the 10-episode gate kept the same
   two failures (tasks 0 and 8) as the untouched policy.
3. The weight buffer is allocated in the autocast dtype (bf16) when autocast is active. The parent
   allocated it in the input dtype (fp32 for the vision merger) and let autocast cast it inside
   `F.linear`; filling bf16 directly applies the same rounding and halves that buffer.

All changes are in place on the evaluator's policy object; no compressed copy is left behind.

## Why
The parent's resident figure was peak-bound: 1.570 GB peak vs 1.118 GB of packed weights, so
~450 MB was transient dequantization storage. Chunking removes that transient without changing a
single weight value (bit-exact outputs), and group 64 removes scale bytes. Both attack the two
terms of `resident = max(walk, live CUDA, peak CUDA)` without touching the zero-margin 8/10 gate.

## Last check output
```
status: ok
memory removed: 79.1% (resident 1173797888 of baseline 5604584960 bytes)
action error (max normalized difference): 0.08203125
latency ms: median 244.10, p95 280.08
```
Run as `python -P -m evaluation.compression.study solution --workspace /tmp/work --phase check`
(final workspace state, `workspace_sha256` ddcf536b114ea8f2a6bf24c266e5d9375e3ae8e8324dce94dab582780ed74e6c).

## Gate smoke rollout (same 10 development episodes the evaluator runs)
```
status: ok, successes: 8/10
size_reduction: 0.7905648906426783
resident_bytes: 1173796864, baseline_resident_bytes: 5604584960
resident_before_inference_bytes: 1057599910
resident_after_inference_bytes: 1057724505
peak_cuda_bytes: 1173796864
```
Failures are tasks 0 and 8, exactly the tasks the untouched policy fails; tasks 1-7 and 9 pass.

## Next
- The remaining gap is 1.174 GB peak vs 1.058 GB of packed weights; the leftover is the full bf16
  weight buffer of the largest layer plus activations. Chunking the GEMM itself would remove it
  (~40 MB, ~0.7 points) but changes accumulation order by ~1 bf16 ulp, so it needs a gate run.
- A 3-bit codebook for the low-influence middle text MLPs (using `calibration["block_influence"]`)
  would save ~130 MB (~2.5 points) but changes weight values materially; validate on the gate.
- Group size 128 saves another ~31 MB but is the next quality risk after group 64.
