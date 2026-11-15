# Proposal (iter_001-b1-a2)

## Change relative to parent (`iter_001-b1-a1`, score 0.7955)

The parent keeps text decoder layers 0-2 and 27 at LLM.int8 and puts layers 3-26 in NF4.
This node removes the last int8 text layers:

1. **Remove the training-only JEPA world model** (`video_encoder`, `video_predictor`,
   `video_processor`) exactly as the parent did.
2. **Compact the text embedding** to the 53 calibrated prompt token ids and drop the unused
   tied LM head, exactly as the parent did.
3. **All 28 text decoder layers now use NF4** (the parent kept 0-2 and 27 at int8).
4. **The vision tower and action head stay NF4**, as in the parent.

Everything is replaced in place on the evaluator's own policy object, so no compressed copy
is left beside the original.

## Why

The parent's 79.6% result was dominated by the four remaining int8 text layers (~0.22 GB).
Moving them to NF4 is the last large, low-risk storage win on the inference path.  The
calibration block influence is dominated by `text.0`-`text.2` (DeepStack injection) and
`text.27`, but the offline action error barely moved (0.0781 -> 0.0820), and the full
10-episode smoke gate still passes with exactly the same two failures as the untouched
policy (tasks 0 and 8, both 280-step timeouts).  The 4-bit kernel is also faster than the
int8 path on these layers (median 147 ms vs 172 ms).

## Rejected experiment

I also built a custom packed 2-bit kernel (group 64, four values per byte, four alternating
least-squares refits of scale/zero) for the 14 lowest-influence text MLPs (layers 3-16).
It reached 83.3% removed (resident 925881998 bytes) and action error 0.229, but the smoke
gate fell to **7/10**: task 4, which passes with the untouched policy, failed.  That variant
was reverted; the submitted solution is the verified 81.3% all-NF4 variant.

## Last check output

`python tools/check.py` on the submitted solution:

```
status: ok
memory removed: 81.3% (resident 1033404854 of baseline 5532276502 bytes)
action error (max normalized difference): 0.0820312
latency ms: median 146.86, p95 186.51
```

Final smoke phase (`evaluation.compression.study solution --phase smoke`):

```
status: ok, successes 8/10
per task: 0: 0/1, 1: 1/1, 2: 1/1, 3: 1/1, 4: 1/1,
          5: 1/1, 6: 1/1, 7: 1/1, 8: 0/1, 9: 1/1
```

The same two tasks fail as for the untouched policy, so this change introduced no new
episode failure.  The resident measurement reproduced exactly (1033404854 bytes) in the
smoke run.

## What I would try next

- A 3-bit (rather than 2-bit) packed kernel for the layers 3-16 MLPs is the obvious next
  step: it saves about 50 MB (roughly 82.3% removed) while keeping 8 levels per weight, so
  it should be much closer to the all-NF4 action error than the failed 2-bit variant.  It
  still needs a fresh 10-episode smoke check because the gate has no margin.
- Alternatively, pack the NF4 weights with a custom group-128 kernel to remove
  bitsandbytes metadata overhead; the expected gain is only ~20-30 MB.
- Do not quantize the action head with LLM.int8: it is far too slow for the episode budget.
- Do not skip whole text layers: even four skipped layers changed actions by 0.29.
