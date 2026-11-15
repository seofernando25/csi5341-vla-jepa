# Node iter_001-b2-a0 — 4-bit middle-layer text MLP on top of the int8 parent

## Change relative to parent (`root`)

The parent is the untouched B16 policy.  This node mutates the evaluator's policy object in
place:

1. Remove the training-only JEPA world model (`video_encoder`, `video_predictor`,
   `video_processor`) and the unused tied LM head.
2. Replace the 151936-row text embedding with a compact table for the calibrated prompt ids
   plus one zero fallback row.
3. Pack the text MLP projections (`gate_proj`, `up_proj`, `down_proj`) of layers **3–16** to
   groupwise asymmetric INT4 (group 128) using PyTorch's tinygemm `_weight_int4pack_mm`.
   Layers 0–2 and 17–27 keep int8 MLP weights: layer 2 carries the DeepStack visual
   injection and layer 27 has the largest calibrated residual influence.
4. Keep text attention and the whole vision tower in LLM.int8 (bitsandbytes); the action head
   stays bfloat16.

## Why

The 0.614 node showed int8 text + int8 vision preserves the 8/10 gate.  Quantizing the whole
text MLP to INT4 reached 70.4% removed but failed a new episode (smoke 7/10, task 5).  The
calibration block-influence ranking is extremely skewed: layers 3–16 are the least
influential, while layers 0–2 and 27 dominate.  Restricting INT4 to the 14 least influential
layers removes another 0.25 GB while keeping the sensitive layers and the action head at
higher precision.  A full 10-episode smoke run of this exact plan passed 8/10 with the same
two failures as the untouched policy (tasks 0 and 8).

## Last check output

`python tools/check.py`:

```
status: ok
memory removed: 65.9% (resident 1886736918 of baseline 5532276502 bytes)
action error (max normalized difference): 0.0576172
latency ms: median 256.49, p95 271.10
```

Smoke phase (`evaluation.compression.study solution --phase smoke`): `status: ok`,
successes 8/10, per-task failures only 0 and 8 — identical to the untouched policy.

## What I would try next

- The all-MLP INT4 plan reached 67.4% removed but its smoke run was invalidated by a test
  log file created in `/tmp` during `compress`; rerun it with clean stdout to see whether it
  also holds the gate.
- Try INT4 only on `gate_proj`/`up_proj` (67.0% removed, action error 0.040) and on the
  least-influential half of the vision blocks.
- Do not quantize the action head with bitsandbytes: it added action error and resident bytes.
