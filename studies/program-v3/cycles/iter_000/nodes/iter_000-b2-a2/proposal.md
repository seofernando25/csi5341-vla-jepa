# Proposal: packed 3-bit/2-bit text projections on top of NF4

## What changed relative to the parent (`iter_000-b2-a1`)

The parent is NF4 everywhere after the exact removals (world model, compact text embedding,
compact action position table).  This node keeps those removals and the NF4 vision tower /
action head, but replaces four text-decoder projections with a custom weight-only packed
format:

- `self_attn.q_proj`, `self_attn.k_proj`, `mlp.gate_proj`: 3-bit, group 64;
- `mlp.up_proj`: 2-bit, group 64.

`self_attn.v_proj` and `mlp.down_proj` stay NF4.  The packed module stores bit-planes (3/8 or
2/8 byte per weight) and one bf16 scale per 64 input channels, dequantizing one output-row
chunk at a time in `forward`.  All changes are in place, so the evaluator's original policy
object owns the same modules and the old storages are freed.  No compressed copy is kept.

## Why

The remaining resident figure is dominated by the text decoder (~0.73 GB of the parent's
1.03 GB weight walk).  The packed projections are half of the text-decoder weights and remove
another ~180 MB relative to NF4.  The choice of which projections to pack was measured on the
80-frame bank: packing `v_proj` or `down_proj` pushes the action error to ~1.0, while packing
q/k/gate/up stays in the same error band as the parent.  The four packed projections are the
largest safe subset; adding `o_proj` raised the 80-frame error to 0.50 and was rejected.

## Last check output

```
status: ok
memory removed: 82.0% (resident 1008648192 of baseline 5604584960 bytes)
action error (max normalized difference): 0.280273
latency ms: median 238.78, p95 256.38
compress: 3.7 s
nonfinite predictions: 0
```

The workspace's own ten-episode smoke run (`evaluation.rollout.run_episode`, tasks 0..9,
development phase, one trial each, the evaluator's seeds) returned **8/10 successes**:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1
task=4 success=1  task=5 success=1  task=6 success=1  task=7 success=1
task=8 success=0  task=9 success=1
TOTAL 8 / 10 [0, 1, 1, 1, 1, 1, 1, 1, 0, 1]
```

The two failures are tasks 0 and 8, exactly the two the untouched policy and the parent fail,
so the success distribution is unchanged while the resident memory fell from the parent's
1.132 GB to 1.009 GB.

## What I would try next

The remaining weights are the NF4 text decoder, NF4 vision tower and NF4 action head.  The
next safe step is the same gate/up treatment for the vision tower's `linear_fc1` and the
action head's `ff.net.0.proj`, which have the same structure as the text gate/up projections
and are worth another ~30 MB.  I would only take it after a fresh ten-episode smoke run: the
`o_proj` experiment shows that a low 10-frame error does not guarantee the 80-frame error or
the episode gate.  A more aggressive 2-bit `k_proj` or 3-bit `o_proj` is possible but not
worth the risk without the smoke result.
