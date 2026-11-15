# Node iter_001-b2-a1 — calibration-ranked mixed 2/3/4-bit packing

## Change relative to parent (`iter_001-b2-a0`)

The parent keeps text layers 0–2 and 17–27 at int8 and packs only the text MLP of layers
3–16 to INT4; text attention and the whole vision tower stay int8, the action head bf16.
This node replaces that uniform split with a calibration-ranked mixed-precision allocation
over the whole inference core:

1. Remove the training-only JEPA world model (`video_encoder`, `video_predictor`,
   `video_processor`) and the unused tied LM head, and compact the 151936-row text
   embedding to the 53 calibrated prompt ids plus one zero fallback row.
2. Text decoder: all 28 layers use bitsandbytes NF4 (4-bit), except the MLP projections of
   layers 3–16, which use HQQ 3-bit packing.  Text attention stays NF4 everywhere.
3. Vision tower: all 24 blocks use NF4, except blocks 12–22, which use HQQ 2-bit packing.
4. Action head: all blocks use NF4, except blocks 3–11, which use HQQ 2-bit packing; a
   zero-element bf16 activation-dtype anchor keeps the timestep encoder in bf16.
5. HQQ's proximal quantizer is vendored into the workspace (`hqq/`, 61 KB, no file over
   1 MiB) so `compress` never imports from `/tmp/repo`.

All changes are in place on the evaluator's own policy object, so no compressed copy is
left beside the original.

## Why

The parent's 65.9% leaves most of the text decoder at int8.  The 0.7955 sibling showed that
NF4 over text layers 3–26 plus the vision tower holds the 8/10 gate with the same two
baseline failures (tasks 0 and 8).  This node starts from that allocation, then spends the
remaining accuracy budget where calibration says it is cheap:

- `text.2` (influence 1575) and `vision.23` (16.2) stay at 4-bit, never 2-bit.
- Text layers 3–16 have influence 4e-5 … 7e-4, so their MLP can take 3-bit while their
  attention stays 4-bit.
- Vision blocks 12–22 have influence 0.05–0.14, so they can take 2-bit.
- Action blocks 3–11 have influence 0.002–0.02, so they can take 2-bit.

Screening on the evaluator's 80 held-out frames (action error, max normalized difference):

| candidate | removed | action error |
|---|---|---|
| parent b2-a0 | 65.9% | 0.0576 |
| b1-a1 NF4 text 3–26 + vision + action | 79.7% | 0.0781 |
| all text NF4 (incl. 0–2, 27) | 81.4% | 0.0781 |
| 3-bit MLP text 3–16, rest NF4 | 82.0% | 0.0957 |
| 2-bit MLP text 3–16, rest NF4 | 83.4% | 0.334 |
| 2-bit whole text 3–16, rest NF4 | 84.0% | 0.977 |
| chosen: 3-bit MLP 3–16 + 2-bit vision 12–22 + 2-bit action 3–11 | 82.8% | 0.1133 |

The chosen candidate is the most aggressive variant that still reproduced the untouched
policy's exact 8/10 smoke pattern.

## Last check output

`python tools/check.py`:

```
status: ok
memory removed: 82.8% (resident 953115990 of baseline 5532276502 bytes)
action error (max normalized difference): 0.113281
latency ms: median 178.91, p95 216.36
```

Final smoke phase (`evaluation.compression.study solution --phase smoke`):

```
status: ok, successes 8/10
per task: 0: 0/1, 1: 1/1, 2: 1/1, 3: 1/1, 4: 1/1,
          5: 1/1, 6: 1/1, 7: 1/1, 8: 0/1, 9: 1/1
resident 953115990 bytes, size reduction 0.827717
```

The same two tasks fail as for the untouched policy (0 and 8, both 280-step timeouts), so
the compression introduced no new episode failure.

## What I would try next

- Push text layers 17–20 from NF4 to 3-bit MLP: the check error was 0.117 for 3-bit MLP
  over 3–26, so this is a small additional gain (~0.3 pp) with a modest error increase.
- Try 2-bit on action blocks 12–15 (influence 0.02 and below) for another ~0.2 pp.
- Do not use 2-bit on the whole text layer or on text attention: 2-bit whole layers gave
  action error 0.977, and 2-bit MLP-only gave 0.334, both far outside the safe band.
- Keep `text.2` and `vision.23` at 4-bit; their influence is two orders of magnitude above
  every other block.
