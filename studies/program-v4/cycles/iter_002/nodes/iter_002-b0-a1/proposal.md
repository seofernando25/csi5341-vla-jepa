# Node iter_002-b0-a1 — compiled flow loop inside the CUDA graph

## What changed relative to the parent (`iter_002-b0-a0`)

The parent copies the evaluation fast path into the workspace, drops the world model and
`lm_head`, and replays one captured CUDA graph per instruction with `use_compile=False`.
Its outputs are bit-identical to the untouched policy.

This node keeps that exact graph and adds one change: the four-step flow-matching loop in
`fastpath.GraphedPolicy._flow` is compiled once with

```python
torch.compile(flow_body, mode="default", fullgraph=True, dynamic=False)
```

and called from inside the captured graph. `flow_body` has fixed shapes for every task
(`tokens` 1x32x2048, `noise` 1x7x7, `state` 1x1x8), so one compiled graph serves all ten
instructions. The vision tower and the 28-layer decoder remain eager and bit-exact; only
the 16-block DiT stack repeated four times is fused.

The flow-matching noise is still drawn eagerly with `torch.randn` on every call, so a seeded
call uses the same noise as the stock path. The compiled flow changes the physical arm
actions by only `9.1e-06` MSE (limit `0.01`) and keeps the perturbation-response correlation
at `0.974` (minimum `0.5`), because the pixel-sensitive vision tower and the decoder are
untouched.

## Why

Profiling the parent graph by stage on task 0 gave: vision blocks 7.58 ms, decoder 12.50 ms,
flow 6.41 ms (sum 26.5 ms). The flow loop is 24 % of the call and is a fixed-shape stack of
small DiT blocks, where eager execution is launch- and pointwise-traffic-bound. Inductor
fuses it into a much shorter kernel sequence: the isolated flow stage drops from 6.41 ms to
4.49 ms in a CUDA graph, about 1.9 ms per call. The decoder is the larger share, but
compiling it (even with the vision tower left eager) changed the perturbation response
enough to fail the guard: check speedup 2.80-3.27x but worst-task correlation 0.35-0.49,
below 0.5. That experiment is why only the flow is compiled here.

## Last check output

`python tools/check.py` (phase `check`, 6+6 interleaved blocks):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 2.8257 (latency_speedup)
speedup: point 2.8904, 95% [2.8257, 2.9111], 6+6 blocks, interleaved
speedup per task: 0 2.882, 1 2.884, 2 2.872, 3 2.925, 4 2.923, 5 2.931, 6 2.860, 7 2.880, 8 2.922, 9 2.828
latency ms: reference median 76.03, candidate median 26.52, p95 28.48
arm action error (worst task): 9.10565e-06 (limit 0.01)
perturbation correlation (worst task): 0.974 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The full smoke phase (`--phase smoke`, the evaluator's 10-episode gate) also passed:

```
status: ok
successes 8 / 10  (tasks 0 and 8 fail, exactly as the untouched reference does)
score 3.0440, 95% [3.0440, 3.2674], 6+6 blocks, interleaved
point 3.1182, per task: 0 3.092, 1 3.148, 2 3.152, 3 3.107, 4 3.121,
                          5 3.119, 6 3.123, 7 3.069, 8 3.053, 9 3.200
reference median 77.0 ms, candidate median 27.44 ms, p95 31.04 ms
reference peak CUDA 6295837696 B, candidate peak 5421900800 B
arm action error 8.57e-06, perturbation correlation 0.974, no guard violations
```

## What I would try next

- Compile only the decoder's MLP (not attention/RMSNorm) or use Inductor settings that keep
  the perturbation response; the full decoder compile was faster but failed correlation.
- Try `mode="max-autotune-no-cudagraphs"` on the flow loop, and pre-build all ten instruction
  plans inside `compress` so no capture happens during the untimed equivalence check.
- Only consider fewer flow steps or a smaller action head if the arm-action error stays well
  under the guard and the smoke gate still passes; the current flow compile is nearly exact,
  which is the safest way to preserve the no-margin 8/10 gate.
