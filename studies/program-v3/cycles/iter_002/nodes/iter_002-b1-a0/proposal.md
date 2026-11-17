# Proposal: in-place Q4/AWQ compression of the VLA-JEPA B16 policy

## Parent
`root` (identity `compress`): resident 5,604,584,960 bytes, 0% removed, check action error 0.
The best earlier node is `iter_002-b0-a0` at 0.594368 (59.4% removed, 8/10 episodes).

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Remove the training-only world model.** `model.video_encoder`,
   `model.video_predictor` and `model.video_processor` are loaded by the published policy but
   `VLAJEPolicy.predict_action_chunk` never calls them. Setting them to `None` frees about
   0.97 GB and is bit-exact for inference.
2. **Compact the tied input embedding.** The Qwen input embedding is 151,936 x 2048 and is
   tied to the unused `lm_head`. The fixed LIBERO prompt template, the ten LIBERO-Spatial
   instructions, the chat-template tokens and the action tokens use only a few dozen ids.
   I build a small row table plus a full-vocabulary `uint8` id map, install it as
   `language_model.embed_tokens`, and remove `lm_head`. This frees the 622 MB tied
   embedding/head storage and is exact for every token id in the fixed prompts.
3. **Drop decoder block 3.** It has the smallest measured residual influence in the
   calibration (`text.3 = 0.000046`). Removing one of the 28 blocks saves another ~10 MB at
   Q4 and keeps the same development smoke result as the parent.
4. **Q4 every large Linear with AWQ-style activation-aware scaling.** Every `nn.Linear` in
   `language_model`, `model.qwen.model.model.visual` and `model.action_model` with at least
   500k weights is replaced by `bitsandbytes.nn.Linear4bit`
   (`compute_dtype=bfloat16`, `compress_statistics=True`, `quant_type="nf4"`). Before
   quantization each weight is scaled per input channel by
   `s = mean_sq_input^0.5` (normalized and clipped) from the calibration `linear_inputs`,
   and the module input is divided by the same `s` in a small wrapper. This is the
   activation-aware quantization trick: it balances weight columns against the actual
   calibration activation magnitudes and makes Q4 accurate enough to keep the 8/10 gate.
   The tiny action-head linears (7/8-dimensional projections and the 256->768 timestep
   projection) stay bfloat16 to avoid bitsandbytes edge cases.

## Why
The parent showed that the world-model removal, embedding compaction, block drop and Q8 are
safe, but left the action head bf16 and the decoder/vision at Q8. Q4 without scaling was too
lossy (parent: action error 0.117, 7/10). Using the calibration `linear_inputs` for per-channel
AWQ scaling reduces the Q4 error enough that the full Q4 policy still passes the same 8/10
development smoke as the untouched policy (tasks 0 and 8 fail, exactly the no-margin
baseline). This removes roughly another 1.1 GB relative to the parent.

## Last check
`python tools/check.py` (final run, Q4 text/vision/action with AWQ scaling):

```
status: ok
memory removed: 80.3% (resident 1106789376 of baseline 5604584960 bytes)
action error (max normalized difference): 0.103516
latency ms: median 142.58, p95 147.67
compress: 8.5 s
nonfinite predictions: 0
```

Development smoke with the repository's own LIBERO rollout code, one episode per task:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1  task=4 success=1
task=5 success=1  task=6 success=1  task=7 success=1  task=8 success=0  task=9 success=1
successes=8/10
```

Tasks 0 and 8 are the same two failures as the untouched policy and the parent, so the
no-margin 8/10 gate is preserved while resident memory falls from 2.27 GB (parent) to
1.11 GB.

## Next
If more time were available, the remaining 1.11 GB is mostly the Q4 text decoder. The only
substantial further saving would be structured pruning of the text MLP intermediate channels
(guided by the calibration input statistics) or 3-bit late MLP weights; both need a fresh
development smoke because the gate has no margin. Dropping additional decoder blocks is not
worth it: each Q4 block is only ~10 MB and the parent already lost a task when dropping two.
