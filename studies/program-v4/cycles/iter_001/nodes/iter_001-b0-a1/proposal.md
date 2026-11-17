# Proposal: compiled CUDA graphs with two flow steps

## Change relative to the parent (`iter_001-b0-a0`)

The parent captured one CUDA graph per instruction and replayed the stock arithmetic exactly
(`use_compile=False`), scoring 2.578460. This node keeps that graph machinery and adds two changes
inside the graph:

1. `fastpath.GraphedPolicy` is constructed with `use_compile=True`, so `torch.compile` fuses the
   elementwise kernels (RMSNorm, RoPE, SiLU, residual adds) and the small GEMMs of the vision
   tower, decoder and action head. The compiled body is still captured into one CUDA graph per
   instruction, so the timed call is one graph replay. `fastpath.py` gains a `compile_kwargs`
   attribute for this.
2. The flow-matching head runs two steps instead of four (`num_inference_timesteps = 2`).

The parent's memory changes stay: the inference-dead V-JEPA world model and the tied vocabulary
output head are dropped, so peak CUDA stays below the reference's.

## Why it passes the guards

Offline probe on 30 bank frames covering all ten tasks (same seed and `Pipeline.sample` batches
as the frozen references), worst task over the ten:

| variant | median ms | worst physical arm MSE | max normalized diff |
|---|---|---|---|
| parent graph (`use_compile=False`) | 26.37 | 0.000000 | 0.000 |
| compile only | 19.39 | 0.000009 | 0.016 |
| **compile + 2 flow steps (this node)** | **18.45** | **0.000042** | **0.027** |
| compile + 2 steps + vision 16-22 + text 3-5 | 14.86 | 0.002781 | 0.258 |

The more aggressive block removal was rejected: its evaluator check scored 4.6785 but failed the
guards (arm error 0.0127 on fresh task-0 frames against 0.01, and perturbation correlations as low
as 0.082 against 0.5). The approximation error was comparable to the small pixel-perturbation
signal. The two-step head changes actions by two orders of magnitude less, so the perturbation
response stays with the reference's (0.928). `torch.compile` changes only bf16 rounding.

The equivalence checks are internal to the candidate and hold because `select_action` delegates to
the same graphed `predict_action_chunk`. No answers are stored and each replay consumes the
current observation and a fresh noise draw.

## Last check output

`/tmp/venv/bin/python tools/check.py` (phase `check`, task 0 episode, all ten tasks timed):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 4.0736 (latency_speedup)
speedup: point 4.0964, 95% [4.0736, 4.1188], 6+6 blocks, interleaved
speedup per task: 0 4.107, 1 4.110, 2 4.098, 3 4.100, 4 4.105, 5 4.129, 6 4.064, 7 4.068, 8 4.098, 9 4.083
latency ms: reference median 74.42, candidate median 18.45, p95 19.23
arm action error (worst task): 4.24779e-05 (limit 0.01)
perturbation correlation (worst task): 0.928 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

Full ten-episode smoke phase (`--phase smoke`, one episode per task):

```
status: ok, successes 8/10, violations []
score: 4.0195 (latency_speedup)
speedup: point 4.0517, 95% [4.0195, 4.0947], 6+6 blocks, interleaved
per-task speedup: 4.027 .. 4.075
latency ms: reference median 74.42, candidate median 18.88, p95 19.55
peak CUDA: candidate 5421900800 B, reference 6295837696 B
arm action error 4.25e-05, perturbation correlation 0.928
per-task successes: tasks 0 and 8 fail, exactly as the untouched reference
```

## What I would try next

- Fuse the host-side input preparation (`to_pixel_values`, image stack, state reshape) into the
  graph's static buffers; it is now a visible fraction of the 18.9 ms call.
- Revisit block removal only with a correction that preserves the perturbation response: prune
  blocks whose contribution is nearly constant across the perturbation, or fine-tune the remaining
  weights against the reference's action and perturbation deltas. The 14.86 ms aggressive variant
  shows the headroom (about 4.7x) if the perturbation guard can be kept.
- Try compile modes (`max-autotune`) once the cache is warm; the first probe showed no gain over
  the default mode.
