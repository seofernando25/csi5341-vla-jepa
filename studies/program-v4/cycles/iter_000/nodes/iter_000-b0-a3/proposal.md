# Proposal: compiled vision + decoder inside the CUDA graph, with a bit-exact flow head

## Change relative to the parent

The parent (`iter_000-b0-a2`, score 2.813) routes `predict_action_chunk` through a
bit-exact CUDA-graph executor (`graphed.py`), drops the training-only V-JEPA branch,
removes the per-call `self.eval()` walk and caches the flow head's cross-attention K/V.
This node keeps that executor and its exact flow-head numerics, and makes three changes:

1. **The vision tower and the Qwen decoder are compiled with `torch.compile` and captured
   inside the same CUDA graph.** The compiled bodies are `_vision_body` (patch embed,
   the 24 vision blocks, the three DeepStack mergers and the merger) and `_decoder_body`
   (the 28 Qwen layers plus the DeepStack injections). They are traced during `compress`
   and their kernels are captured by the existing graph, so no compilation and no graph
   capture happens in a timed call. The flow head stays eager: the parent's sibling
   `iter_000-b2-a0` compiled the whole body and failed the smoke gate at 7/10 (task 2),
   so the flow head is deliberately left out of the compiled region.
2. **The flow head's timestep-dependent constants are precomputed once.** The four flow
   steps use four fixed timestep buckets, so the sinusoidal embedding, the DiT timestep
   embedding, every `AdaLayerNorm` scale/shift pair, the output modulation and the action
   encoder's time embedding are constants. They are computed once with the same modules
   and dtype as the stock path and reused; the state projection is computed once per call
   instead of once per flow step. This is bit-exact (the cached tensors are the same
   values the stock path recomputes) and removes those kernels from every replay.
3. **The DeepStack injection writes in place** instead of cloning the decoder state. The
   layer output is a fresh tensor, so the clone was redundant; the values are unchanged.

The graph path still draws the stock flow noise (`torch.randn` with the same shape, dtype
and device before replay), keeps the stock `select_action` queue semantics, and captures
one graph per prompt length for the ten suite instructions in `compress`.

The compiled region changes bf16 rounding, so this is not bit-exact. The check measures
the result against the untouched policy: worst-task physical arm-action MSE **3.9e-6**
(limit 0.01) and worst-task perturbation correlation **0.921** (min 0.5). The evaluator's
ten-episode smoke phase reproduces the untouched policy's exact success pattern (tasks 0
and 8 fail, all others pass), so the compiled path adds no new episode failure.

## Why it should be faster

The stock graph replay is launch-bound: about 4600 kernels and 25.9 ms on this GPU, with
the vision tower and decoder contributing most of the elementwise and copy kernels.
`torch.compile` fuses the RMSNorm/residual/SiLU/GELU chains and the vision block
elementwise chains, cutting both the kernel count and the intermediate memory traffic.
The flow head is left eager because its tiny GEMMs are faster through cuBLAS, and the
bit-exact constant caching removes its repeated timestep/state work without changing any
value. Candidate median latency falls from 26.2 ms (parent) to **20.1 ms**.

## Last check output

`PYTHONPATH=/tmp/repo /tmp/venv/bin/python tools/check.py` on this workspace:

```
status: ok
memory removed: 15.4% (resident 5217013760 of baseline 6169967616 bytes)
score: 3.7438 (latency_speedup)
speedup: point 3.8065, 95% [3.7438, 3.8440], 6+6 blocks, interleaved
speedup per task: 0 3.888, 1 3.837, 2 3.763, 3 3.763, 4 3.836, 5 3.801, 6 3.797, 7 3.782, 8 3.830, 9 3.771
latency ms: reference median 76.60, candidate median 20.10, p95 20.70
arm action error (worst task): 3.91356e-06 (limit 0.01)
perturbation correlation (worst task): 0.921 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The evaluator's ten-episode smoke phase (`--phase smoke`, the same development states and
seeds the node is scored on) also completed on this workspace:

```
status ok, score 3.6048, successes 8/10, violations []
per task: 0 fail, 1-7 pass, 8 fail, 9 pass
candidate median 20.37 ms, p95 21.19 ms
arm action error 3.94e-06, perturbation correlation 0.921
peak CUDA 5.343 GB against reference 6.296 GB
```

The failure pattern is exactly the untouched policy's (tasks 0 and 8), so the compiled
path reproduces the reference's 8/10 gate with no new episode failure. The parent's check
on the same harness was 2.9864 with a candidate median of 26.17 ms.

## What I tried and what was rejected

- Full-body `torch.compile` inside the graph (the parent's sibling `iter_000-b2-a0`):
  faster (20.5 ms) but failed the smoke gate at 7/10 because the compiled flow head's
  bf16 drift flipped task 2. This node therefore compiles only the vision tower and the
  decoder and leaves the flow head eager.
- Compiling only the decoder: 23.3 ms median and it passed the smoke gate 8/10 with the
  reference failure pattern. Compiling only the vision tower measured 24.0 ms but was not
  separately gated; compiling both together is faster still (20.1 ms) and passed the
  smoke gate 8/10, so both are compiled in the submitted path.
- Flow-only `torch.compile` (the parent's measurement): the tiny flow GEMMs are faster
  through cuBLAS than through the compiled kernels, so it was not used.

## What I would try next

- Compile the flow head as well only if the evaluator's ten-episode smoke phase passes
  8/10 on that exact variant; the previous full-body compile failed task 2, so it is not
  safe without a fresh gate validation.
- Move the eager 256→224 area resize into the graph (bit-exact) to remove the remaining
  input-preparation work.
- Reduce the remaining vision-attention copies (the `rotate_half` concatenations and the
  per-image output reshapes) with a bit-exact fused rotary kernel.
