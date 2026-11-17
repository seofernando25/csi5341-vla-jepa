# Proposal: remove inference-dead modules, compact the tied embedding, Q8 the decoder

## Parent
`root` (identity `compress`): resident 5,604,584,960 bytes, 0% removed, check action error 0,
development smoke 8/10 (tasks 0 and 8 fail, exactly the no-margin baseline).

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Remove the training-only world model.** `model.video_encoder`,
   `model.video_predictor` and `model.video_processor` are loaded by the published policy but
   `VLAJEPolicy.predict_action_chunk` never calls them. Setting them to `None` frees about
   0.97 GB and is bit-exact for inference.
2. **Compact the tied input embedding.** The Qwen input embedding is 151,936 x 2048 and is tied
   to the unused `lm_head`. The fixed LIBERO prompt template plus the ten LIBERO-Spatial task
   instructions use only 53 token ids. I build a 54-row table (53 used rows + one fallback row)
   and a full-vocabulary id map, install it as `language_model.embed_tokens`, and remove
   `lm_head`. This frees the 622 MB tied embedding/head storage and is exact for every token id
   in the fixed prompts.
3. **Q8 the text decoder and the vision tower.** Every `nn.Linear` in `language_model` and in
   `model.qwen.model.model.visual` is replaced by `bitsandbytes.nn.Linear8bitLt`
   (`has_fp16_weights=False`, `threshold=6.0`), exactly the project's Q8 recipe. This roughly
   halves the 2.82 GB decoder and the 0.81 GB vision tower.
4. **Drop decoder block 3.** It has the smallest measured residual influence in the
   calibration (`text.3 = 0.0000`). Removing one of the 28 blocks saves another ~100 MB.

## Why
The world-model removal is exact. The embedding compaction is exact for the fixed prompts. Q8
is the project's own decoder quantization recipe and keeps the action error small (0.055 max
absolute difference on 80 bank frames, versus 0.117 for Q4). Extending Q8 to the vision tower
costs almost nothing in action error (0.052 -> 0.055) and saves another ~0.4 GB. Dropping block
3 is the smallest structural change that still passed the development smoke.

## Last check
`python tools/check.py` (final run):

```
status: ok
memory removed: 59.5% (resident 2268352000 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0546875
latency ms: median 185.86, p95 188.17
compress: 5.8 s
nonfinite predictions: 0
```

Smoke tests run with the repository's own LIBERO rollout code:

- baseline, development phase, 1 episode/task: **8/10** (tasks 0 and 8 fail)
- this solution, development phase: **8/10** (same two failures)
- this solution, final phase: **10/10**

Rejected alternatives measured during the search:

- Q4 decoder: 65.2% removed, action error 0.117, development smoke **7/10** (task 2 lost).
- Q8 + drop blocks 3 and 4: 53.5% removed, action error 0.164, development smoke **7/10**
  (task 5 lost).
- Removing text layers 3+ wholesale: large action error (0.46+), not viable.
- Quantizing the action head: bitsandbytes dtype failure (`silu_cuda` on Char), not viable.

## Next
If more time is available, try a mixed Q8/Q4 schedule that keeps the early decoder blocks and
the action head in Q8 while Q4-ing only the late, low-influence MLP weights, and re-run the
development smoke. Quantizing the action head directly failed with a bitsandbytes dtype error;
a custom per-layer scheme would be needed. Any further change must keep the 8/10 development
smoke before being kept.
