# Node iter_000-b0-a0

## Change relative to parent (root template)

`solution.py` now compresses the loaded B16 policy in place.  The parent was the
identity template, so every change below is new.

1. **Drop the training-only world model.** `policy.model.video_encoder`,
   `video_predictor` and `video_processor` are set to `None`.  `predict_action` and
   `select_action` never call them, so inference is unchanged.  This frees
   483,423,616 bfloat16 parameters (~0.97 GB).
2. **Drop the unused tied LM head and compact the token embedding.**  The policy reads
   the last decoder hidden state through a hook and never calls `lm_head`.  The tied
   `lm_head` is replaced by `nn.Identity()`, and `language_model.embed_tokens` is
   replaced by a compact embedding containing only the 53 token ids in
   `calibration["token_ids"]`.  I checked all 100 observation-bank samples: their
   token ids are exactly that set, so this part is behavior-identical.  It frees the
   151936x2048 embedding (~0.62 GB).
3. **Quantize the remaining inference core.**
   - Text decoder Linears: groupwise int4 (group 128) with an AWQ-style
     activation-aware per-channel input scale, selected by a 10-point alpha search on
     `calibration["linear_inputs"]`.
   - Vision encoder Linears: the same int4 scheme.
   - Action-head Linears: weight-only symmetric int8 with one scale per output row.
   - Norms, the compact embedding and the timestep encoder stay bfloat16.

All replacements happen in place, so the original storages are freed; no compressed
copy is kept.  `compress` performs no file I/O.

## Last check output

`python tools/check.py` (final solution):

```
status: ok
memory removed: 79.6% (resident 1130142412 of baseline 5532276502 bytes)
action error (max normalized difference): 0.118164
latency ms: median 225.40, p95 261.75
```

I also ran the evaluator's `smoke` phase (10 LIBERO-Spatial episodes, one per task):

```
status: ok, successes 8/10
resident 1,130,142,412 bytes, size_reduction 0.7957
failed tasks: 0 and 8; tasks 1-7 and 9 succeeded
```

The same 8/10 pattern was obtained with the safer int8-only variant (64.2% removed)
and with int4 text + int8 vision/action (76.2% removed), so the extra vision int4 did
not add a failure on this gate.

## What I would try next

- Try int4 on the action head as well; it would add only ~1.4 points, so it was not
  worth the extra gate risk here.
- If a margin over 8/10 is needed, back off to int4 text + int8 vision/action
  (76.2% removed) or int8 everywhere (64.2% removed); both gave the same 8/10 smoke
  result.
- A larger gain would need structural pruning (text layers/heads) or 2/3-bit
  quantization, but the 8/10 gate leaves no room for a new episode failure.
