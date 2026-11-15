# Node iter_001-b0-a0 — exact inference-dead tensor removal

## What changed relative to parent (`root`)

`solution.py` now mutates the loaded policy in place:

1. **Remove the JEPA world model** (`policy.model.video_encoder`,
   `video_predictor`, `video_processor`).  `VLAJEPAPolicy.predict_action_chunk`
   only runs `_encode_qwen` and `action_model.predict_action`; the world model is
   used solely by the training loss (`_world_model_loss`).  `_prepare_model_inputs`
   assembles `videos` only when `training=True`.  Removing it is therefore exact.
2. **Drop the tied LM head** (`policy.model.qwen.model.lm_head = nn.Identity()`).
   The policy calls the inner `Qwen3VLModel` and hooks the last decoder block
   (`_qwen_last_decoder_hidden`), so the 151936×2048 LM head is never executed.
3. **Compact the text embedding** to the 53 calibrated prompt token ids (plus all
   tokenizer special ids and action-token ids).  LIBERO-Spatial has ten fixed
   instructions and the calibration covers two observations from every task, so
   every token id that can appear in an episode is retained.  Retained rows are the
   original bfloat16 rows; unseen ids fall back to an appended zero row rather than
   raising.  Because the LM head is tied to this table, replacing the embedding and
   dropping the head together releases the full 622 MB vocabulary storage.

No weight is quantised, pruned, or otherwise modified.  The forward hooks used to
audit the inference path showed that the only parameterised modules never called
are exactly the world model and the LM head.

## Why

The gate has no margin (the untouched policy passes about 8/10), so the safest way
to score is to remove only tensors the inference path provably never reads.  The
world model contributes ~967 MB and the tied vocabulary table ~622 MB of the
5.53 GB baseline.

## Last check output

`python tools/check.py`:

```
status: ok
memory removed: 28.7% (resident 3944067606 of baseline 5532276502 bytes)
action error (max normalized difference): 0
latency ms: median 86.71, p95 108.49
```

A `smoke` rollout (10 LIBERO-Spatial episodes, one per task) also completed with
`status: ok`, 8 successes / 10 episodes — the same two tasks (0 and 8) fail as for
the untouched policy, so the compression introduced no new failure.

## Next

The exact-safe removal is exhausted: a full forward-hook audit found no other
unused parameterised module, and all remaining weights are on the inference path.
The next step, if a higher score were needed, would be a lossy quantisation of the
text decoder, but that would risk the zero-margin 8/10 gate and should only be
attempted with a paired multi-seed rollout comparison.
