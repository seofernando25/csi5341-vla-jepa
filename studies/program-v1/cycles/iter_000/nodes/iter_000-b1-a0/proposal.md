# Proposal: dead-state removal + INT8 text/vision linears for VLA-JEPA B16

## What changed relative to the parent

The parent is the untouched workspace template (`compress` returns the policy unchanged).
This node makes three in-place changes to the loaded B16 policy:

1. **Remove the V-JEPA world model.** `policy.model.video_encoder`, `video_predictor` and
   `video_processor` are set to `None`. `VLAJEPAModel.predict_action` calls
   `_encode_qwen(..., need_action_tokens=False)` and `action_model.predict_action`; it never
   touches the world model. `_prepare_model_inputs(..., training=False)` also does not build
   `videos`. This removes 325.97 M encoder parameters + 157.45 M predictor parameters
   (about 0.967 GB of bf16 storage) with identical outputs.
2. **Compact the token embedding and drop the unused output head.** The full Qwen embedding
   has 151,936 rows (622.3 MB). The frozen calibration contains the 53 unique token ids that
   the ten LIBERO-Spatial prompts and the chat template actually use. The embedding is
   replaced by a 53-row embedding plus a mean-vector fallback row and an int32 lookup table.
   `lm_head` is tied to the same storage and is never called because the policy runs the
   inner `Qwen3VLModel`, so it is replaced by `nn.Identity()`. This removes about 0.621 GB.
3. **INT8-quantize every `nn.Linear` in the text decoder and the vision tower** with
   bitsandbytes `Linear8bitLt` (`has_fp16_weights=False`, threshold 6.0). The original bf16
   weights are released as each module is replaced. The action head and all norms/embeddings
   stay bf16. This halves the remaining text/vision Linear storage (about 1.76 GB before
   quantization).

The final measured resident size is **2,135,266,238 bytes** against a baseline of
**5,532,276,502 bytes**, i.e. **61.4% memory removed**.

## Why this should keep the 8/10 gate

The first two changes are inference-exact: every weight that participates in the forward
pass is bit-identical, and every token id seen in evaluation maps to the same row. That exact
variant scored 28.7% and passed the 10-episode smoke rollout 8/10, with the same two failing
tasks as the untouched policy (tasks 0 and 8).

The INT8 text/vision quantization is the only approximate part. It leaves the action head in
bf16 and keeps the same 8/10 smoke result: tasks 1-7 and 9 succeed, tasks 0 and 8 fail, exactly
as the exact variant and the untouched policy do. The offline check's maximum normalized
action difference is 0.0303, small enough that the ten development episodes did not change
outcome. The gate has no margin, so this is the main risk of the node; the exact 28.7% variant
is the safe fallback if the scorer sees a different episode.

## Last check output

`python tools/check.py` on this workspace:

```
status: ok
memory removed: 61.4% (resident 2135266238 of baseline 5532276502 bytes)
action error (max normalized difference): 0.0302734
latency ms: median 425.30, p95 528.70
```

Smoke rollout (`evaluation.compression.study solution --phase smoke`), 10 development
episodes, one per task:

```
successes: 8 / 10
per task: 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:1
```

The exact structural-only variant (no INT8) also gave 8/10 with the same per-task outcomes
and 28.7% memory removed. The text-only INT8 variant gave 47.8% and 8/10. This node keeps the
text+vision INT8 variant at 61.4%.

## What I would try next

- The action-head INT8 variant reached 64.2% but raised median latency to about 872 ms and
  was not smoke-validated; it is not used here. If latency is not a scoring concern, validate
  it with a smoke rollout before adopting.
- Try 4-bit (NF4) text/vision linears for roughly 75% total reduction, but only if a smoke
  rollout still keeps 8/10; the 0.0303 action error at INT8 is already close to the tolerance
  implied by the zero-margin gate.
- If a different episode set is used, prefer the exact 28.7% variant: it is bit-identical and
  has no quantization risk.
