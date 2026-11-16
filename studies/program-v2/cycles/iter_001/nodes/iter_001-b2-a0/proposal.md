# iter_001-b2-a0 — remove inference-dead weights and compact the prompt embedding

## What changed relative to the parent (`root`)

The parent is the untouched B16 policy (identity `compress`).  This node changes
`solution.py` to make three in-place, inference-exact changes:

1. **Remove the V-JEPA world model.** `policy.model.video_encoder`,
   `video_predictor` and `video_processor` are only used by `_world_model_loss`
   during training.  `predict_action_chunk` / `select_action` call
   `VLAJEPAModel.predict_action`, which uses only Qwen and the action head.
   This drops ~967 MB of bf16 weights (326M + 157M parameters).
2. **Remove the tied `lm_head`.** `_qwen_last_decoder_hidden` calls the inner
   `Qwen3VLModel` directly and never builds logits.  `lm_head.weight` aliases
   `embed_tokens.weight`, so replacing it with `nn.Identity()` is required before
   the embedding can shrink; it frees the full 151936×2048 table.
3. **Compact `embed_tokens`.** The frozen calibration provides the unique prompt
   token ids seen on the fixed LIBERO-Spatial tasks.  A lookup table maps those
   ids to a small embedding table; unseen ids would make `F.embedding` fail
   loudly instead of silently remapping.  This is exact for the fixed task suite
   and reduces the 622 MB embedding to a few MB.

All originals are freed in place, so the evaluator's held reference to the same
policy sees the smaller tensors.  A forward-hook trace of
`predict_action_chunk` confirmed that the only parameterized modules never
called are `lm_head`, the V-JEPA encoder/predictor, and
`action_model.future_tokens` (the latter is read directly as
`future_tokens.weight`, not through its `forward`, so it is retained).

## Last check output

`python tools/check.py` (final run, solution.py unchanged):

```
status: ok
memory removed: 28.4% (resident 4014329344 of baseline 5604584960 bytes)
action error (max normalized difference): 0
latency ms: median 97.75, p95 172.27
```

The evaluator's 10-episode smoke phase on the same workspace:

```
status: ok, phase: smoke
successes: 8 / 10
size_reduction: 0.28374208819202196
resident_bytes: 4014328320, baseline_resident_bytes: 5604584960
peak_cuda_bytes: 4014328320
```

The 8/10 smoke result matches the untouched policy exactly (tasks 0 and 8 fail,
all others succeed), and the action error on all 80 held-out frames is exactly
zero.

## What I would try next

Because the smoke gate is exactly 8/10 (no margin), I deliberately stopped at
exact structural removal rather than adding a lossy step.  If a later node has
evidence of margin (e.g. the untouched policy scores 9–10/10 on the scoring
episodes), the next step would be calibration-aware int8 quantization of the
vision encoder and text decoder projections, keeping the action head in bf16,
with the 80-frame action error and a smoke run as the acceptance test.  A
smaller exact win still available is compacting the action head's
`position_embedding` from 1024 rows to the 7 positions actually used (~1.6 MB),
but it is negligible relative to the 28.4% already removed.
