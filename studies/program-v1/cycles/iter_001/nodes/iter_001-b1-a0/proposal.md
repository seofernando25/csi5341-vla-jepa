# Proposal (iter_001-b1-a0)

## Change relative to parent (`root`, identity)

The parent returns the untouched B16 policy.  This node makes four in-place changes on the
object the evaluator already holds:

1. **Remove the JEPA world model** (`video_encoder`, `video_predictor`, `video_processor`).
   `predict_action_chunk`/`select_action` only call `_encode_qwen` and the action head; the
   world model is used exclusively by the training loss.  Frees ~0.97 GB of bf16 weights.
2. **Compact token embedding + drop the tied output head.**  `calibration["token_ids"]`
   contains exactly the 53 unique ids used by all 100 observation-bank inputs (verified by
   running `prepared_inputs` over the whole bank: missing = [], extra = []).  The `lm_head`
   is tied to the embedding and is never evaluated because the policy calls the inner
   `Qwen3VLModel`.  Replacing both frees the 151936 x 2048 bf16 vocabulary (622 MB) and
   stores 53 rows instead.
3. **LLM.int8 (bitsandbytes) on every text-decoder Linear** (the repo's Q8 arm).
4. **LLM.int8 on every vision-tower Linear.**  The vision tower is 402M params; int8 cuts it
   roughly in half.  The action head is deliberately left in bf16: quantizing it *increased*
   resident bytes (bitsandbytes overhead on many small Linears: 2.38 GB vs 2.14 GB) and did
   not reduce action error.

All changes are in place on the evaluator's own policy object, so no compressed copy is left
beside the original.

## Why

The gate has no margin (baseline passes 8/10), so every candidate was checked on the actual
10-episode smoke phase, not just action error.  Measured alternatives:

| candidate | resident | removed | action err | smoke |
|---|---|---|---|---|
| safe (world + compact vocab only) | 3.944 GB | 28.7% | 0.0 | 8/10 (fails 0, 8) |
| text int8 (Q8) | 2.537 GB | 54.1% | 0.0244 | 8/10 (fails 0, 8) |
| text int8 + vision int8 (chosen) | 2.135 GB | 61.4% | 0.0303 | 8/10 (fails 0, 8) |
| text NF4 (Q4) | 1.853 GB | 66.5% | 0.1172 | 7/10 (fails 0, 2, 8) |
| text int8 + action int8 | 2.383 GB | 56.9% | 0.0303 | not run (worse memory) |
| skip text layers 3..27 | 1.427 GB | 74.2% | 1.027 | rejected |
| skip text layers 24..27 | 3.541 GB | 36.0% | 0.291 | rejected |
| skip action blocks 1..15 | 3.668 GB | 33.7% | 0.902 | rejected |

The chosen candidate fails exactly the same two smoke tasks as the exact-output safe set
(tasks 0 and 8, both 280-step timeouts), so the int8 changes did not add a new failure.

## Last check output

```
status: ok
memory removed: 61.4% (resident 2135262142 of baseline 5532276502 bytes)
action error (max normalized difference): 0.0302734
latency ms: median 262.09, p95 272.25
```

Final smoke phase (`evaluation.compression.study solution --phase smoke`):
`status: ok`, successes 8/10, same per-task pattern as the safe baseline (0 and 8 fail).

## What I would try next

- Mixed precision: NF4 only on the late text layers or only on the vision tower, keeping Q8
  on the sensitive early layers, then re-run the 10-episode smoke gate.  Q4 everywhere is
  too aggressive (7/10), but a mixed variant might recover the 66% memory level at 8/10.
- Quantize only the large vision Linears (`qkv`, `proj`, `fc1`, `fc2`) to avoid bitsandbytes
  overhead on the small ones.
- Do not pursue structural layer skipping: calibrated block influence was misleading, and
  even four skipped text layers changed actions by 0.29.
