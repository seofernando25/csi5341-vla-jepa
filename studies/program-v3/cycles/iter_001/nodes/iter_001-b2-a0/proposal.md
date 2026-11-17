# iter_001-b2-a0 — low-bit the whole inference stack, drop the dead world model

## Change relative to the parent (`root`)

The parent workspace is the untouched template (`compress` returns the policy unchanged).
This node makes in-place changes to the evaluator's policy object:

1. **Delete the JEPA world model.** `policy.model.video_encoder` (V-JEPA2 ViT-L, ~0.652 GB
   bf16), `policy.model.video_predictor` (~0.315 GB bf16) and `video_processor` are set to
   `None`; `enable_world_model` is cleared on both configs.  `predict_action_chunk` only calls
   `VLAJEPAModel.predict_action`, which uses `_encode_qwen` and `VLAJEPAActionHead`; the world
   model is used only by the training `forward`, so this is bit-exact for scored actions.
2. **Drop the tied `lm_head`.**  The checkpoint ties `lm_head.weight` to
   `embed_tokens.weight`; inference calls the inner `Qwen3VLModel`, which never uses `lm_head`.
   Setting it to `None` removes the second reference so the bf16 token table can be freed.
3. **Int8 the token embedding.**  `embed_tokens` becomes `bitsandbytes.nn.Embedding8bit`
   (one absmax scale per row, bf16 output), replacing the 0.622 GB bf16 table with ~0.311 GB.
4. **Low-bit every `nn.Linear` used at inference:**
   - language-decoder attention projections (`q/k/v/o_proj`): `Linear8bitLt`
     (`has_fp16_weights=False`, threshold 6.0);
   - language-decoder MLP projections (`gate/up/down_proj`): `Linear4bit` NF4
     (`compute_dtype=bfloat16`, double quantization);
   - Qwen visual tower (104 linears) and action head (123 linears): NF4.
   Original bf16 modules are released as they are replaced; no copy is kept.  The action
   head's tiny timestep encoder stays bf16 because it reads
   `next(self.parameters()).dtype` and would otherwise cast its sinusoidal input to an
   integer dtype.

`compress` mutates the object the evaluator already holds and returns it, then calls
`gc.collect()` and `torch.cuda.empty_cache()`.

## Why

The world model is ~0.97 GB of the ~5.53 GB of bf16 weights and contributes nothing at
inference.  The remaining weights are almost entirely Linear leaves: language decoder
2.82 GB, visual tower 0.81 GB, action head 0.31 GB.  Int8 halves those; NF4 quarters the
parts that tolerate it.  The sibling `iter_001-b1-a0` removed the world model and int8'd
only the language decoder (0.4051).  This node additionally removes the visual/action/
embedding bytes.  NF4 was applied component by component and checked on the actual
development episodes, not just action error, because the 8/10 gate has no margin.

## Last check output

```
status: ok
memory removed: 71.1% (resident 1620711936 of baseline 5604584960 bytes)
action error (max normalized difference): 0.081543
latency ms: median 173.11, p95 175.47
compress: 10.8 s
nonfinite predictions: 0
```

## Episode verification (evaluator-style, development trial 0, tasks 0–9)

- world model + int8 language decoder + int8 visual/action/embedding: **8/10**, failures
  tasks 0 and 8 (55.6% removed).
- same, with language-decoder MLP changed to NF4: **8/10**, same failures.
- final recipe (NF4 language MLP + NF4 visual tower + NF4 action head, int8 language
  attention and int8 embedding): **8/10**, same failures (tasks 0 and 8), 71.1% removed.
- The untouched policy and `iter_001-b1-a0` also fail exactly tasks 0 and 8, so the final
  recipe preserves the baseline success set and meets the 8/10 gate.

## What I would try next

- NF4 the language-decoder attention projections too (about another 0.3 GB).  The sibling
  `iter_001-b1-a0` found all-NF4 decoder cost task 2, so this needs an episode run before
  keeping it; the attention path is the sensitive part.
- NF4 the token embedding (about another 0.155 GB) with the same episode check.
- Do not trade away the verified 8/10 set for action error alone: the gate has no margin.
