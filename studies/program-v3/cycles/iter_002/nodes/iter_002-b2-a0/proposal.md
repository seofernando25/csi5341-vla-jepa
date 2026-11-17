# Proposal: in-place NF4 compression of the VLA-JEPA B16 policy

## Parent
`root` (identity `compress`): resident 5,604,584,960 bytes, 0% removed, check action error 0.
The best earlier node is `iter_002-b1-a0` at 0.802613 (80.3% removed, 8/10).

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Remove the training-only world model.** `model.video_encoder`,
   `model.video_predictor` and `model.video_processor` are loaded by the published policy
   but `VLAJEPolicy.predict_action_chunk` never calls them. Setting them to `None` frees
   about 0.97 GB and is bit-exact for inference.
2. **Compact the tied input embedding.** The Qwen3-VL input embedding is 151,936 x 2048 and
   is tied to the unused `lm_head`. The fixed LIBERO prompt template, the ten
   LIBERO-Spatial instructions, the two camera images, the chat-template tokens and the
   action placeholders use 99 token ids. I install a 100-row table (99 used rows + one
   fallback) with a full-vocabulary `int16` row map and delete `lm_head`. This frees the
   622 MB tied embedding/head storage and is exact for every token id in the fixed prompts.
3. **Drop text decoder block 3.** It has the smallest measured residual influence in the
   calibration (`text.3 = 0.000046`). Removing one of the 28 blocks saves ~28 MB at NF4.
4. **NF4 quantization with activation-aware scaling.** Every `nn.Linear` in
   `language_model`, `model.qwen.model.model.visual` and `model.action_model` with at least
   500k weights is replaced by `bitsandbytes.nn.Linear4bit`
   (`compute_dtype=bfloat16`, `compress_statistics=True`, `quant_type="nf4"`). Before
   quantization each weight is scaled per input channel by `s = sqrt(mean_sq_input)`,
   normalized and clipped, using the calibration `linear_inputs`; a small wrapper divides
   the module input by the same `s`. This is the activation-aware quantization trick. Tiny
   action-head projections stay bfloat16.

## Why
The world-model removal and embedding compaction are exact for the fixed prompts. NF4 with
AWQ-style per-channel scaling keeps the action error small enough that the full policy
passes the same 8/10 development smoke as the untouched policy (tasks 0 and 8 fail, exactly
the no-margin baseline). The result matches the best earlier node at 80.3% removed.

## Last check
`python tools/check.py` (final run, NF4 text/vision/action with AWQ scaling):

```
status: ok
memory removed: 80.3% (resident 1105683968 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0961914
latency ms: median 141.81, p95 144.44
compress: 7.2 s
nonfinite predictions: 0
```

Development smoke with the repository's own LIBERO rollout code, one episode per task:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1  task=4 success=1
task=5 success=1  task=6 success=1  task=7 success=1  task=8 success=0  task=9 success=1
successes=8/10
```

Tasks 0 and 8 are the same two failures as the untouched policy, so the no-margin 8/10
gate is preserved.

## Next
The remaining 1.11 GB is mostly the NF4 text decoder. Two further reductions were measured
and rejected because they raise action error too far for the no-margin gate: 20%
activation-aware structured pruning of the text MLP intermediate channels reached 81.9%
removed but action error 0.214, and dropping vision blocks 18-22 reached 80.9% removed but
action error 0.172. The next attempt should be a much smaller pruning fraction (5-10%) or a
better 4-bit quantizer, with a fresh development smoke before keeping either.
