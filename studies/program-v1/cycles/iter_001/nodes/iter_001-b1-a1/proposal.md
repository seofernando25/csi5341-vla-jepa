# Proposal (iter_001-b1-a1)

## Change relative to parent (`iter_001-b1-a0`)

The parent removes the JEPA world model and the tied vocabulary head, compacts the text
embedding, and applies LLM.int8 to every text-decoder Linear and every vision-tower Linear.
This node keeps those exact removals but changes the bit allocation to the measured block
sensitivity:

1. **Text decoder layers 0-2 and 27 stay LLM.int8.**  The calibration's block-influence
   measurement is dominated by `text.0`, `text.1`, `text.2` and `text.27`; these are the
   layers whose perturbations matter most.
2. **Text decoder layers 3-26 use NF4.**  These 24 low-influence layers hold most of the
   decoder's 1.4 GB of int8 weights.  NF4 halves that storage.
3. **The vision tower uses NF4** instead of LLM.int8, saving a further ~0.2 GB.
4. **The action head uses NF4 with `compress_statistics=False`** and a zero-element bf16
   anchor on its timestep encoder.  LLM.int8 on the action head was rejected: it took ~3 s
   per prediction (vs ~150 ms for NF4) and retained extra storage.  NF4 is both faster and
   slightly lower action error in the offline screen.

All changes are in place on the evaluator's own policy object, so no compressed copy is left
beside the original.

## Why

The gate has no margin (the untouched policy passes about 8/10), so the bit allocation follows
the calibration's block influence rather than a uniform scheme.  The parent's uniform int8
scored 61.4%.  Offline screening of alternatives on the evaluator's 80 held-out frames gave:

| candidate | action err | latency |
|---|---|---|
| parent: text int8 + vision int8 | 0.0303 | 326 ms |
| text 3-26 NF4 + vision int8 | 0.105 | 171 ms |
| text 3-26 NF4 + vision NF4 | 0.0859 | 117 ms |
| text 3-26 NF4 + vision NF4 + action NF4 (chosen) | 0.0781 | 153 ms |
| text 2-27 NF4 + vision NF4 (no action NF4) | 0.0977 | 125 ms |

The chosen candidate has both the best official memory result and the lowest action error
among the 4-bit variants, because the high-influence text ends stay at int8.  Quantizing
`text.2` or `text.27` was deliberately avoided: those are the two largest text block
influences after the outlier at `text.0`.

## Last check output

`python tools/check.py`:

```
status: ok
memory removed: 79.6% (resident 1131170374 of baseline 5532276502 bytes)
action error (max normalized difference): 0.078125
latency ms: median 162.85, p95 207.80
```

Final smoke phase (`evaluation.compression.study solution --phase smoke`):

```
status: ok, successes 8/10
per task: 0: 0/1, 1: 1/1, 2: 1/1, 3: 1/1, 4: 1/1,
          5: 1/1, 6: 1/1, 7: 1/1, 8: 0/1, 9: 1/1
```

The same two tasks fail as for the untouched policy (0 and 8, both 280-step timeouts), so the
compression introduced no new episode failure.  A second identical smoke run reproduced
8/10 with the same per-task pattern and the same 1131170374-byte resident measurement.

## What I would try next

- The remaining memory is mostly the 24 NF4 text layers and the NF4 vision tower.  A custom
  packed INT4 kernel (the repo's `PackedInt4`) could remove bitsandbytes metadata overhead,
  but it needs a fresh 10-episode smoke check before it is worth the risk.
- Try text layers 2-27 at NF4 (keeping only 0-1 and 27 at int8) for another ~1 percentage
  point, but only with a paired smoke run: `text.2` has a very large measured influence.
- Do not use LLM.int8 on the action head: it is far too slow for the episode budget.
