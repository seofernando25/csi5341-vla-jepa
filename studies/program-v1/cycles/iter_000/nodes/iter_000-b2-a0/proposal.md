# Node iter_000-b2-a0

## Change relative to parent (root template)

The parent was the identity template. `solution.py` now mutates the loaded B16 policy
in place and returns it; no compressed copy is retained.

1. **Drop the training-only V-JEPA world model.** `policy.model.video_encoder`,
   `video_predictor` and `video_processor` are set to `None`. `predict_action` and
   `select_action` never call them, so inference is unchanged. This frees
   483,423,616 bfloat16 parameters (~0.97 GB).
2. **Drop the unused tied LM head and compact the token embedding.** The policy reads
   the last decoder hidden state through a hook and never calls `lm_head`, so it is
   replaced by `nn.Identity()`. `language_model.embed_tokens` is replaced by a
   53-row embedding (the token ids in `calibration["token_ids"]`) plus one mean
   fallback row and an int32 lookup. This frees the 151936x2048 embedding (~0.62 GB).
3. **Quantize the remaining inference core.**
   - Text decoder Linears: groupwise int4 (group 128) with an AWQ-style per-channel
     input scale chosen by a 10-point alpha search on `calibration["linear_inputs"]`,
     using PyTorch's Ampere tinygemm int4 kernel.
   - Vision encoder Linears: the same packed int4 scheme.
   - Action-head Linears: the same packed int4 scheme, except the tiny
     `action_encoder.layer1` interface whose 7-wide input cannot use 128-wide int4
     groups; that one stays row-wise symmetric int8.

The only difference from the best previous node (`iter_000-b0-a0`, 79.57%) is that
this node also packs the action head to int4 instead of int8. That removes another
~77 MB. The text/vision int4 recipe is the one that already passed 8/10.

All replacements happen in place, so the original storages are freed. `compress`
performs no file I/O and keeps no calibration tensors on the GPU.

## Last check output

`python tools/check.py` on the final workspace:

```
status: ok
memory removed: 80.6% (resident 1074000318 of baseline 5532276502 bytes)
action error (max normalized difference): 0.150391
latency ms: median 193.04, p95 313.88
```

Smoke rollout (10 LIBERO-Spatial episodes, one per task) on this exact recipe:

```
status: ok, successes 8/10
failed tasks: 0 and 8; tasks 1-7 and 9 succeeded
```

Tasks 0 and 8 are the same two failures as the untouched policy, so this is the
8/10 gate with the same per-task pattern as the parent baseline.

## Variants tried and rejected

- **Action head int8 (the b0 recipe):** 79.6% removed, action error 0.117, smoke
  8/10. Safe, but lower score.
- **Action head int4:** 80.6% removed, action error 0.150, smoke 8/10. This is the
  deployed recipe.
- **Prune 4 low-influence text layers (3-6):** 82.5% removed but action error
  0.592; not smoke-tested because the error is far outside the safe range.
- **Prune 14 low-influence text layers (3-16):** 87.4% removed, action error 1.016;
  rejected.
- **HQQ 3-bit text+vision:** 82.9% removed, action error 2.012; rejected.
- **HQQ 2-bit on text layers 3-16:** 83.4% removed, action error 0.858; rejected.
- **HQQ 3-bit on text layers 3-16 + 3-bit vision:** 82.0% removed, action error
  0.163, but the smoke rollout failed task 4 (7/10), so it was reverted. The extra
  0.013 action error over the deployed recipe was enough to flip a task.

The block-influence ranking was misleading for this policy: layers with tiny mean
residual influence still matter for the embodied-action token positions, and
2/3-bit HQQ on them raises the action error sharply.

## What I would try next

- Smoke-test **vision-only 3-bit HQQ** with text/action int4. It should land near
  81.9% and may keep the 8/10 pattern; the combined text+vision 3-bit variant was
  only 0.013 action-error above the deployed recipe but failed task 4, so the vision
  part alone is the remaining cheap gain.
- Use the 20 calibration states to collect real per-layer input covariances and run
  GPTQ/error-compensated int4 or int3 instead of weight-only AWQ. That is the most
  promising route to 85%+ without the layer-pruning error seen here.
- If a larger structural gain is needed, prune text layers only with a
  distillation/correction step on the calibration bank; pruning alone is not viable
  at this gate.
