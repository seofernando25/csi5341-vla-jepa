# Node iter_000-b2-a0 — exact trims + AWQ-scaled packed INT3 text decoder

Parent: `root` (identity `compress`, score 0.0). Siblings: `iter_000-b0-a0` (exact trims +
INT8 text, 0.527) and `iter_000-b1-a0` (exact trims + INT4 text/vision/action, 0.797).
This node keeps the same exact inference-path trims but stores the 2.82 GB text decoder at
**3 bits** instead of 4, with AWQ-style per-input-channel scaling and a packed
10-values-per-int32 layout. Vision and action stay on the proven INT4 tinygemm path.

## What changed relative to the parent

**Exact (the deployed inference path never touches these):**

1. Drop the JEPA world model: `video_encoder` (652 MB), `video_predictor` (323 MB),
   `video_processor`. `VLAJEPAModel.predict_action` only calls `_encode_qwen` +
   `action_model.predict_action`.
2. Compact token embedding: the fixed prompt template produces only 53 distinct ids
   (`calibration["token_ids"]`, two observations per task, so all 10 LIBERO-Spatial
   instructions are covered). The 151936x2048 bf16 table (622 MB) becomes those rows
   (0.2 MB) plus a 151936-entry int32 lookup (0.6 MB). `lm_head` is tied to the full table
   and is never called (`_qwen_last_decoder_hidden` runs the inner `Qwen3VLModel`, not the
   causal-LM wrapper), so it is replaced by `nn.Identity()` before the table is swapped.

**Approximate (weight-only; activations stay bf16):**

3. Text decoder (2.82 GB, 196 linears) → packed **INT3**, group 128, per-group bf16
   scale+zero, 10 values packed per int32. Per-layer AWQ input scale from
   `calibration["linear_inputs"]` (5-point alpha search on activation-weighted
   reconstruction error). ~0.57 GB.
4. Vision tower (806 MB) and action head (308 MB) → **INT4** tinygemm, group 128,
   AWQ input scale (10-point alpha search); tiny linears (<64K elements, e.g. the 7x768
   decoder) stay bf16. ~0.30 GB.

All surgery is in place on the same policy object; the old modules are dropped and
`gc.collect()` + `torch.cuda.empty_cache()` run before returning. No file is opened or
written inside `compress`, and no bf16 copy of a quantized weight is kept.

## Last check output (`python tools/check.py`)

```
status: ok
memory removed: 81.0% (resident 1064860160 of baseline 5604584960 bytes)
action error (max normalized difference): 0.256836
latency ms: median 272.25, p95 375.22
```

## Real gate evidence (evaluator's own smoke phase, same development seeds/states)

```
python -m evaluation.compression.study solution --workspace /tmp/work --phase smoke --out /tmp/out/smoke_int3
{"status": "ok", "phase": "smoke"}
successes: 8 / 10,  size_reduction: 0.8100021422460514
per_task: 0:0/1, 1:1/1, 2:1/1, 3:1/1, 4:1/1, 5:1/1, 6:1/1, 7:1/1, 8:0/1, 9:1/1
```

The two failures (tasks 0 and 8) are the same tasks the untouched policy and the INT4
sibling fail; every passing episode finishes well inside the 280-step budget. The
evaluator measured resident before inference 933.1 MB, after inference 933.3 MB, peak
1064.9 MB — the peak is the binding figure and is the packed weights plus the normal
forward activation working set.

## Rejected experiment

A more aggressive mixed variant (2-bit middle text blocks, 4-bit first/last) reached
84.1% removed but action error rose to 0.961 and the real smoke gate scored **0/10**
(all ten episodes failed). It was reverted; the 3-bit solution above is the final one.

## Why this is not hiding memory

Nothing is parked, deferred or reloaded. The world model and dead embedding rows are
dropped and collected, `lm_head` is unregistered, and the packed INT3/INT4 tensors are the
resident representation. The only transient is one dequantized text layer inside its
`forward`, which is freed immediately after the matmul. No workspace file is over 1 MiB
(workspace total is about 22 KB).

## Next

- The remaining ~0.93 GB is mostly 0.57 GB of packed INT3 text weights plus the forward
  activation peak. A real further win needs 2-bit text, but the mixed 2-bit experiment
  shows the policy is past its accuracy cliff there; a safer next try is 2-bit only for
  the middle **MLP** projections while keeping middle attention at 3 bits and the
  first/last blocks at 4 bits.
- A cheaper +1-2 points: double-quantize the INT3 group scales/zeros (about 44 MB of bf16
  metadata) or put the vision patch-embed and remaining norms in int8.
- If a future evaluator is less deterministic, the INT4 sibling (0.797, action error
  0.115) is the conservative fallback; it is not kept in this workspace.
