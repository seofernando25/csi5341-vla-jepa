# Node iter_001-b2-a2 — 3-bit text MLP 3–26 with group-128 HQQ metadata

## Change relative to parent (`iter_001-b2-a1`, score 0.8277)

The parent removes the training-only JEPA world model, drops the unused tied LM head,
compacts the text embedding to the 53 calibrated prompt ids, and applies a calibration-ranked
mixed 2/3/4-bit allocation: NF4 text, HQQ 3-bit text MLP 3–16, HQQ 2-bit vision 12–22 and
action 3–11.  This node keeps those structural removals and the vision/action allocation,
and makes two changes:

1. **The 3-bit text-MLP range is extended from layers 3–16 to layers 3–26.**  Text attention
   and the sensitive MLP layers 0–2/27 stay at NF4.  This is the parent's suggested next
   step: the parent measured 3-bit MLP over 3–26 at action error 0.117 (vs 0.096 for 3–16),
   so the extra ten layers are cheap.
2. **The HQQ group size is 128 instead of 64.**  Scale and zero metadata are stored once per
   group, so this halves that overhead on every packed layer (text MLP, 2-bit vision blocks
   12–22, 2-bit action blocks 3–11) without changing the number of weight levels.

All changes are in place on the evaluator's own policy object, so no compressed copy is left
beside the original.

## Rejected experiments

- **Remove text layers 3–16 as identity blocks.**  Their calibration residual influence is
  only 4.6e-5 … 6.8e-4, two orders of magnitude below layers 17–26, and no earlier node had
  isolated that range.  The check reached 88.8% removed (617.6 MB resident) but action error
  was **1.011**.  The influence average is dominated by image tokens; the embodied-action
  tokens do depend on those blocks.  Reverted.
- **All text Linears at 3-bit (including attention).**  Check: 83.6% removed, action error
  **1.586**.  Text attention is far more sensitive than the MLP.  Reverted.
- **Group 64 vs 128.**  Group 64 with the same bit plan gave 83.1% removed and action error
  0.146; group 128 gives 83.8% and 0.184.  Both passed the 10-episode gate, so the smaller
  group-128 footprint was kept.
- **Vision blocks 3–11 at 2-bit.**  Extending the 2-bit range from 12–22 to 3–22 reached
  84.3% removed but action error jumped to 0.284; adding action blocks 12–15 as well gave
  84.4% and 0.284.  The vision blocks 3–11 are the cause, so the parent's 12–22 range was
  kept.

## Last check output

`python tools/check.py` on the submitted group-128 solution:

```
status: ok
memory removed: 83.8% (resident 897751902 of baseline 5532276502 bytes)
action error (max normalized difference): 0.183594
latency ms: median 236.10, p95 295.26
```

The final default-plan check reproduced the resident measurement exactly
(897751902 bytes) after the group-128 setting was written into `solution.py`.

Final smoke phase (`evaluation.compression.study solution --phase smoke`):

```
status: ok, successes 8/10
per task: 0: 0/1, 1: 1/1, 2: 1/1, 3: 1/1, 4: 1/1,
          5: 1/1, 6: 1/1, 7: 1/1, 8: 0/1, 9: 1/1
resident 897751902 bytes, size reduction 0.837724686812843
```

The same two tasks fail as for the untouched policy (0 and 8, both 280-step timeouts), so
the compression introduced no new episode failure.  The group-64 variant of the same plan
also passed 8/10 at 932938590 bytes (0.831364), so the group-128 change is gate-safe on this
run.

## What I would try next

- Try group 256 on the text MLP only (≈14 MB more) while keeping group 128 on vision/action,
  then re-check and re-run smoke.
- Move only action blocks 12–15 to 2-bit (≈5 MB).  The combined vision+action experiment
  showed the error came entirely from the vision extension, so this small action change is
  likely safe, but it needs its own smoke run.
- Do not extend 2-bit vision below block 12: blocks 3–11 raised action error to 0.284.
- Do not lower text attention below NF4: 3-bit attention alone gave action error 1.59.
- Do not remove text layers 3–16: their mean influence is misleading because it is dominated
  by image tokens.
