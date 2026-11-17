# iter_001-b1-a0 — drop the dead world model, int8 the language decoder

## Change relative to the parent (`root`)

The parent workspace is the untouched template (`compress` returns the policy unchanged).
This node makes two in-place changes to the evaluator's policy object:

1. **Delete the JEPA world model.** `policy.model.video_encoder` (V-JEPA2 ViT-L, ~0.652 GB
   bf16), `policy.model.video_predictor` (~0.315 GB bf16) and `video_processor` are set to
   `None`; `enable_world_model` is cleared on both configs.  `VLAJEPAPolicy.predict_action_chunk`
   only calls `VLAJEPAModel.predict_action`, which runs `_encode_qwen` (Qwen3-VL visual tower +
   language decoder) and `VLAJEPAActionHead`.  The world model is used only by the training
   `forward`, so this is bit-exact for the scored actions.
2. **Quantize the Qwen language decoder to int8.** Every `nn.Linear` under
   `policy.model.qwen.model.model.language_model` is replaced by a
   `bitsandbytes.nn.Linear8bitLt` (`has_fp16_weights=False`, threshold 6.0), the project's Q8
   recipe.  The original bf16 modules are released as they are replaced; no copy is kept.
   Embeddings, the visual tower, the action head and `lm_head` stay bf16.

`compress` mutates the object the evaluator already holds and returns it, then calls
`gc.collect()` and `torch.cuda.empty_cache()`.

## Why

The world model is ~0.97 GB of the ~5.53 GB of bf16 weights and contributes nothing at
inference.  Int8 is the largest decoder reduction that still matched the baseline on the
actual episodes: NF4 (the project's Q4 recipe, 54.1% removed) failed one extra episode, while
int8 preserved the baseline's exact success set.

## Last check output

```
status: ok
memory removed: 40.5% (resident 3332854272 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0244141
latency ms: median 149.20, p95 152.08
compress: 4.9 s
nonfinite predictions: 0
```
(Final `solution.py`, just re-run.)

## Episode verification (evaluator-style, development trial 0, tasks 0–9)

- world-model removal only: **8/10** (failures on tasks 0 and 8)
- world-model removal + int8 decoder: **8/10**, same failures (tasks 0 and 8)
- world-model removal + NF4 decoder: **7/10** (task 2 additionally failed) — rejected

The 8/10 gate is met by the int8 node with the same margin as the untouched policy.

## What I would try next

- Int8-quantize the Qwen visual tower and/or the action head as well (about +0.55 GB, ~50%
  total) and re-run the 10 episodes; keep only if the success set is unchanged.
- Activation-aware mixed precision: keep the first three decoder blocks (deepstack
  injection) and the final block bf16, NF4 only the low-influence middle MLPs, then verify
  episodes before keeping it.
- NF4 is too lossy for this no-margin gate despite the larger 54.1% reduction.
