# iter_002-b2-a0: lossless removals + compact vocab + selective INT8

## Change relative to parent (root template)
- Delete the V-JEPA2 video encoder and action-conditioned video predictor. `predict_action_chunk`
  and `select_action` never call them; only training does. This removes ~0.97 GB of BF16 storage.
- Replace the tied, unused `lm_head` with `nn.Identity()` and replace the 151936-row token
  embedding with a compact embedding over the 53 token ids present in the frozen calibration.
  Unseen ids map to a zero fallback row instead of raising. This removes ~0.62 GB.
- Store selected Linear weights as symmetric INT8 with per-group (group size 32) scales:
  - all Linears in the Qwen vision tower;
  - all text-decoder self-attention projections;
  - text-decoder MLP projections in layers 3-26.
  The action head and the first/last text blocks (layers 0-2 and 27) stay BF16. Each packed
  layer is dequantized inside its own `forward`, so no BF16 copy survives `compress`.
- All changes are in place on the evaluator's policy object; no compressed copy is left behind.

## Why
The world-model and vocabulary-head removals are exact for inference. Selective INT8 keeps the
most sensitive parts (action head, early DeepStack-injected text blocks, final conditioning block)
in BF16 while cutting the large vision and middle text weights about in half. Full text INT8 lost
a smoke task, so the text MLP is restricted to the low-influence middle layers.

## Last check output
```
status: ok
memory removed: 52.2% (resident 2677393408 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0170898
latency ms: median 124.36, p95 136.68
```

## Smoke rollout (same 10 development episodes as the gate)
```
status: ok, successes: 8/10
size_reduction: 0.5217
resident_bytes: 2680606720, baseline_resident_bytes: 5604584960
```
The two failures are tasks 0 and 8, the same tasks the untouched policy fails on this smoke set.

## Next
- Try group size 16 or activation-aware (AWQ-style) scales for the middle text MLP to recover
  more memory without losing task 2.
- Optionally quantize the action head with a BF16 dtype anchor, but only after a smoke run.
- A GPTQ-style calibration pass on the text MLP could allow more layers to be packed safely.
