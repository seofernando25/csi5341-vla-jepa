# iter_002-b1-a0: remove inference-dead state, then INT8/NF4 the live towers

Parent: `root` (identity `compress`, score 0.0).

## Change relative to the parent

All changes are in place on the loaded policy so the original storages are released.

1. **Drop the V-JEPA world model.** `policy.model.video_encoder`,
   `video_predictor` and `video_processor` are used only by the training-time
   world-model loss. The inference path (`_encode_qwen` -> DiT action head)
   never calls them. Setting the three attributes to `None` removes ~0.92 GiB
   of BF16 weights with no effect on actions.
2. **Compact the tied input embedding.** The decoder only embeds the token ids
   that occur in the prompt; the frozen `calibration["token_ids"]` covers all
   ten LIBERO-Spatial instructions plus the fixed action/embodied tokens. The
   151936 x 2048 BF16 `embed_tokens` matrix is replaced by a table with only
   those rows, and the unused tied `lm_head` is replaced by `nn.Identity()`
   so the original matrix is actually freed. Unseen ids map to a zero row
   instead of raising, so an unexpected token cannot crash a rollout.
3. **INT8 text decoder.** Every `nn.Linear` in the Qwen language model is
   stored as per-group symmetric INT8 (group 64) with BF16 scales and
   dequantized at call time. This halves the 3.44 GiB decoder weight storage.
4. **NF4 vision tower and action head.** Every `nn.Linear` in the Qwen visual
   tower and the DiT action head (except the tiny timestep encoder and
   sub-64-wide projections) is stored with bitsandbytes NF4 and BF16 compute.
   The decoder stays INT8 because full NF4 on the decoder was measurably worse
   (see below).

The returned object is the same policy instance, so `select_action` and
`predict_action_chunk` keep working.

## Why

The untouched policy scores 0 by construction. The world model and LM head are
provably dead at inference, so removing them is free. The remaining live
weights are then quantized only where a smoke rollout still passes the 8/10
gate. The decoder is the most sensitive component, so it gets the gentler INT8
treatment; the vision tower and action head tolerate NF4 on the fixed smoke
episodes.

## Last check output

`cd /tmp/work && /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 68.3% (resident 1753640790 of baseline 5532276502 bytes)
action error (max normalized difference): 0.0947266
latency ms: median 143.66, p95 201.73
```

Smoke rollout of this candidate (`--phase smoke`, 10 episodes, one per task):
**8/10 successes**, failures on tasks 0 and 8. The untouched baseline was run
through the same evaluator on the same seeds and also scored **8/10** with the
same two failures, so this candidate does not lose a baseline episode. A second
smoke run of the final candidate reproduced **8/10** with the same two failures.

Rejected intermediate: NF4 on the text decoder as well reached 66.5% memory
removed but only **7/10** smoke successes (task 2 also failed), which would
score 0. The final candidate keeps the decoder INT8 and pushes NF4 only into
the vision tower and action head, reaching 68.3% while preserving 8/10.

## Next

- The repeat smoke run reproduced 8/10 with the same two failures, so the
  candidate is stable on the fixed development seeds.
- Try mixed INT8/NF4 per decoder layer (attention INT8, MLP NF4) or GPTQ on
  the decoder to move past 68% without touching the sensitive attention path.
- If the gate is ever given margin, quantize the timestep encoder and the
  sub-64-wide action projections too (small additional savings).
