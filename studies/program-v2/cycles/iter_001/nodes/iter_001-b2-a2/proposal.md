# iter_001-b2-a2 — measured attempts and final safe fallback

## What changed relative to the parent (`iter_001-b2-a1`)

The final `solution.py` keeps the parent's exact INT4 weight-only recipe: remove the
V-JEPA world model, drop the tied `lm_head`, compact the prompt embedding, compact the
action position table, and INT4-pack every large Linear in the text, vision and action
roots.  I explored two more aggressive directions and rejected both on evidence:

1. **Skip near-identity text decoder blocks.**  Calibration `block_influence` reports
   text layers 3–26 at ~1e-4 relative residual energy, so I replaced them with identity
   blocks.  This reached 91.3% memory removed, but action error exploded to 1.61 and the
   smoke run would not be trustworthy.  One-at-a-time ablation showed even single-block
   removal costs 0.12–0.55 action error; removing layers 3 and 8 together gave 0.119.
2. **INT2 MLP weights.**  I implemented groupwise affine 2-bit packing (group 64) for
   the text and vision MLP matrices, keeping attention and the action head at INT4.
   `tools/check.py` reported 83.1% memory removed and action error 0.516.  The evaluator
   smoke phase then scored **6/10** (tasks 1 and 4 flipped to failure), so this variant
   was rejected and reverted.

The submitted file therefore reproduces the parent's known-safe 8/10 behavior.  The
workspace still differs from the parent by this proposal and the exploratory code kept
in `solution.py`; the active `MODE` is `int4_all` and `KEEP_TEXT_LAYERS` is all 28
layers, so no block is skipped.

## Last check output

Final submitted file is the parent INT4 recipe; the last measured check for that exact
recipe (parent node `iter_001-b2-a1`) was:

```
status: ok
memory removed: 79.6% (resident 1142731264 of baseline 5604584960 bytes)
action error (max normalized difference): 0.140625
latency ms: median 165.61, p95 259.28
smoke: 8 / 10 (tasks 0 and 8 fail, as in the untouched policy)
```

The exploratory INT2 check (not submitted) measured:

```
status: ok
memory removed: 83.1% (resident 947134976 of baseline 5604584960 bytes)
action error: 0.515625
smoke: 6 / 10 — rejected
```

## What I would try next

The smoke gate has no margin, so only changes that preserve the exact per-task success
pattern are viable.  The next experiment should be a smaller INT2/INT3 scope, e.g.
2-bit only on the text MLP of layers 3–26 while keeping layers 0–2, 27 and the action
head at INT4, with a smoke run as the acceptance test.  A safer alternative is INT4
group size 64 for the text MLP (slightly more memory, lower error), or removing only
layer 3 (single-block action error 0.137, essentially the parent's 0.141) for a small
exact-ish win.
