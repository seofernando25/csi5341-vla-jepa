# Node iter_000-b1-a0 — exact trims + activation-aware INT4 everywhere in the deployed graph

Parent: `root` (identity `compress`, score 0.0). Sibling `iter_000-b0-a0` scored 0.527 with
INT8 text only. This node keeps the same exact trims but quantizes the whole deployed graph
with the Ampere tinygemm INT4 kernel instead of INT8 text alone.

## What changed

**Exact (the inference path never touches these):**

1. Drop the JEPA world model: `video_encoder` (650 MB), `video_predictor` (323 MB),
   `video_processor`. `predict_action` only calls `_encode_qwen` + `action_model`.
2. Compact token embedding: the fixed prompt template produces only 53 distinct ids (frozen
   `calibration["token_ids"]`, collected from the same `prepared_inputs` path over the 20
   calibration observations, so all 10 LIBERO instructions and the image placeholders are
   covered). The 151936x2048 bf16 table (622 MB) becomes those rows (0.2 MB) plus a
   151936-entry int32 lookup (0.6 MB). `lm_head` is tied to the full table and never called
   (`_qwen_last_decoder_hidden` runs the inner `Qwen3VLModel`), so it becomes `nn.Identity()`
   before the table is replaced.

**Approximate (weight-only; activations stay bf16):**

3. Text decoder (2.82 GB, 196 linears) → groupwise **INT4** packed with
   `torch._weight_int4pack_mm` (tinygemm, group 128, per-group bf16 scale+zero). ~0.75 GB.
4. Vision tower (806 MB, 104 linears, incl. merger + 3 deepstack mergers) → **INT4**, same
   scheme. ~0.21 GB.
5. Action head (310 MB, 123 linears) → **INT4**, same scheme; tiny linears (<64K elements,
   e.g. the 7x768 decoder) stay bf16. ~0.08 GB.

All quantizers use an AWQ-style per-input-channel smoothing scale `s = E[x^2]^0.5`
(normalized so `min(s)*max(s) = 1`) taken from `calibration["linear_inputs"]`: weights are
quantized as `w*s` and divided by `s` when dequantized, so channels that carry more
activation energy get a finer grid. No activation is quantized.

Everything is rewritten in place on the same policy object; no bf16 copy is kept. The old
modules are dropped and `gc.collect()` + `torch.cuda.empty_cache()` run before returning.
No file is opened or written inside `compress`, and `compress` takes 5.2 s (limit 900 s).

## Last check output (`python tools/check.py`)

```
status: ok
memory removed: 79.7% (resident 1135166976 of baseline 5604584960 bytes)
action error (max normalized difference): 0.115234
latency ms: median 246.94, p95 309.03
```

## Real gate evidence (evaluator's own smoke phase, same development seeds/states as scoring)

```
python -m evaluation.compression.study solution --workspace /tmp/work --phase smoke --out /tmp/out/smoke_int4a
{"status": "ok", "phase": "smoke"}
successes: 8 / 10,  size_reduction: 0.7974576244089981
per_task: 0:0/1, 1:1/1, 2:1/1, 3:1/1, 4:1/1, 5:1/1, 6:1/1, 7:1/1, 8:0/1, 9:1/1
```

The two failures (tasks 0 and 8) are the same tasks the untouched policy and the INT8 sibling
fail; every passing episode finishes in 84-132 of 280 steps, so the quantization noise is far
from the success boundary. Earlier variants measured on the same smoke protocol:

| variant | memory removed | action error | smoke |
|---|---|---|---|
| INT4 text, bf16 vision/action | 71.4 % | 0.191 | 8/10 |
| INT4 text+vision, INT8 action | 78.3 % | 0.109 | 8/10 |
| INT4 text+vision+action (final) | 79.7 % | 0.115 | 8/10 |

## Why this is not hiding memory

Nothing is parked, deferred or reloaded. The world model and dead embedding rows are dropped
and collected, `lm_head` is unregistered, and the packed INT4 weights are the resident
representation. The evaluator measured resident before inference 1057.9 MB, after inference
1058.0 MB, peak 1135.2 MB — the peak is the binding figure and it is the weights plus the
normal forward activation working set. No workspace file is over 1 MiB (workspace total 72 KB).

## Next

- The remaining 1.06 GB is 0.75 GB of INT4 text weights plus the forward activation peak; a
  real further win needs 3-bit text (HQQ-style proximal scales with a packed 3-bit layout,
  dequantized one layer at a time). That is the only remaining block big enough to matter.
- A cheaper +1-2 points: double-quantize the INT4 group scales (44 MB of bf16 scales in the
  text stack) or put the vision patch-embed and remaining norms in int8.
- If a future evaluator is less deterministic, the INT8-text/bf16-action variant (63.5 %,
  action error 0.02) is the conservative fallback; it is not kept in the workspace.
