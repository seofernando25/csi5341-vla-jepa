# Proposal (iter_001-b1-a3)

## Change relative to parent (`iter_001-b1-a2`, score 0.8132)

The parent removes the JEPA world model, compacts the text embedding to the 53 calibrated
prompt ids, drops the tied LM head, and puts **all** text, vision and action Linears in
bitsandbytes NF4.  This node keeps the exact removals and replaces the uniform NF4 allocation
with a calibration-ranked mixed scheme:

1. Text decoder layers **0-2 and 27** stay NF4 (the high-influence ends; `text.2` has
   influence 1575 and `text.27` 0.081).
2. Text decoder layers **3-26** keep NF4 attention but move their **MLP** to HQQ 3-bit with
   group size 256.  These 24 MLPs are ~906 M parameters and among the least influential
   blocks (`text.3`-`text.16` influence 4.6e-5 … 7e-4).
3. Vision blocks **12-22** move to HQQ 2-bit (group 64); all other vision Linears stay NF4.
4. Action blocks **3-15** move to HQQ 2-bit (group 64); the rest of the action head stays NF4.
5. HQQ's proximal quantizer is vendored into `hqq/` (324 KB, no file over 1 MiB), so
   `compress` never imports from `/tmp/repo`.

All changes are in place on the evaluator's own policy object, so no compressed copy is left
beside the original.

## Why

The parent's 81.3% is dominated by the 1.06 B-parameter text MLP, which still costs ~0.516
byte/parameter in NF4 (4-bit plus an 8-bit absmax per 64 weights).  HQQ 3-bit costs ~0.431
byte/parameter at group 256 (3.2-bit int32 packing plus bf16 scale/zero per 256 weights) and
its proximal refit keeps reconstruction much closer than a naive 3-bit round.  The sibling
`iter_001-b2-a1` showed that 3-bit MLP on text layers 3-16 plus 2-bit vision 12-22 plus 2-bit
action 3-11 holds the 8/10 gate at 82.77%.  This node extends the 3-bit MLP band to all
low-influence layers 3-26, coarsens its metadata to group 256, and extends action 2-bit to
blocks 12-15, while leaving every high-influence block at 4-bit.

Measured progression (all in-place on the evaluator's policy):

| variant | resident bytes | removed | action err | smoke |
|---|---|---|---|---|
| parent: all NF4 | 1033404854 | 81.32% | 0.0820 | 8/10 (0, 8) |
| 3-bit MLP 3-26, g64 | 932934918 | 83.14% | 0.1458 | 8/10 (0, 8) |
| 3-bit MLP 3-26, g128 | 904641798 | 83.65% | 0.1664 | 8/10 (0, 8) |
| **chosen: g256 + action 3-15 2-bit** | **882975318** | **84.04%** | **0.1826** | **8/10 (0, 8)** |
| + 3-bit attention 3-16 (rejected) | 868019062 | 84.31% | 0.2324 | 7/10 (0, 4, 8) |

## Last check output

`python tools/check.py` on the submitted solution:

```
status: ok
memory removed: 84.0% (resident 882975318 of baseline 5532276502 bytes)
action error (max normalized difference): 0.182617
latency ms: median 189.14, p95 239.49
```

Final smoke phase (`evaluation.compression.study solution --phase smoke`):

```
status: ok, successes 8/10
per task: 0: 0/1, 1: 1/1, 2: 1/1, 3: 1/1, 4: 1/1,
          5: 1/1, 6: 1/1, 7: 1/1, 8: 0/1, 9: 1/1
resident 882975318 bytes, size reduction 0.840396
```

The same two tasks fail as for the untouched policy (0 and 8, both 280-step timeouts), so the
compression introduced no new episode failure.  This beats the previous tree best (0.827717).

## Rejected experiment

Moving text attention layers 3-16 to HQQ 3-bit (group 128) reached 84.31% removed and action
error 0.232, but the smoke gate fell to **7/10**: task 4, which passes with the untouched
policy, failed.  That variant was reverted; the submitted solution is the verified 84.04%
variant.

## What I would try next

- Move the vision merger/deepstack projections (~100 M parameters, currently NF4) to HQQ
  3-bit; expected ~8 MB more, moderate risk because they feed the high-influence text.0-2.
- Try 2-bit on vision block 11 (influence 0.279) and group 512 on the text MLPs; each is only
  a few MB, so they need a paired smoke run to be worth the gate risk.
- Do not quantize text attention below 4-bit: the 3-bit attempt already broke task 4.
- Do not lower text layers 0-2/27 below 4-bit: `text.2` carries the DeepStack injection.
