# Proposal: CUDA-graph execution of the stock VLA-JEPA forward

## Change relative to the parent (`root`)

The parent was the untouched B16 policy. `solution.py` now:

1. drops the inference-dead V-JEPA world model (`video_encoder`, `video_predictor`,
   `video_processor`) and the unused tied vocabulary output head (`lm_head`), which the policy
   never calls because it invokes the inner `Qwen3VLModel`;
2. constructs `fastpath.GraphedPolicy` (copied from `evaluation/fastpath.py` into the workspace,
   with its NF4 helper copied too) with `use_compile=False, dense=True`, and routes
   `policy.predict_action_chunk` through it. `select_action` already delegates to
   `predict_action_chunk` when its action queue is empty, and the evaluator calls `reset()` before
   every timed call, so every timed call is one graph replay.

The graphed path:

- precomputes the instruction-dependent constants (text embeddings, position ids, rotary tables,
  visual/action index tables, patch grid) once per instruction;
- runs the vision tower, decoder, DeepStack injection and the four flow-matching steps without the
  host synchronizations in the stock mask/attention code;
- copies the current images, state and freshly drawn flow noise into static buffers and replays one
  captured CUDA graph per instruction.

No weights are changed, no work is moved to `reset`, the CPU or another process, no files are
opened during `compress`, and no answers are stored: each replay consumes the current observation
and a fresh noise draw.

## Why it should pass the guards

`use_compile=False` keeps the graph's arithmetic identical to the stock eager path. On 18 bank
frames covering all ten tasks (local probe, same seed and `Pipeline.sample` batches) the maximum
absolute action difference against the stock policy was exactly `0`. The smoke run confirms this:
worst-task arm-action error `0`, perturbation correlation `1.000`, and the same 8/10 episode
successes as the reference (tasks 0 and 8 fail for the reference too). Dropping the world model and
`lm_head` cannot change actions because neither is on the inference path; it only lowers peak CUDA
memory, which the reference still pays for.

## Last check output

`python tools/check.py` (phase `check`, task 0 episode, all ten tasks timed):

```
status: ok
memory removed: 14.2% (resident 5296030720 of baseline 6169967616 bytes)
score: 2.6244 (latency_speedup)
speedup: point 2.6453, 95% [2.6244, 2.6717], 6+6 blocks, interleaved
speedup per task: 0 2.688, 1 2.646, 2 2.662, 3 2.598, 4 2.680, 5 2.640, 6 2.617, 7 2.662, 8 2.619, 9 2.643
latency ms: reference median 73.92, candidate median 28.15, p95 29.92
arm action error (worst task): 0 (limit 0.01)
perturbation correlation (worst task): 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

Full ten-episode smoke phase (`--phase smoke`, one episode per task):

```
status: ok, successes 8/10, violations []
score: 2.6010 (latency_speedup)
speedup: point 2.6318, 95% [2.6010, 2.6591], 6+6 blocks, interleaved
per-task speedup: 2.606 .. 2.659
latency ms: reference median 73.92, candidate median 28.15
peak CUDA: candidate 5421900800 B, reference 6295837696 B
arm action error 0, perturbation correlation 1.000
```

## What I would try next

- Measure `use_compile=True` (kernel fusion) and keep it only if the arm-error, perturbation and
  success guards still pass; the current bit-exact path is the safe fallback.
- Fuse the remaining host-side input preparation (`to_pixel_values`, image stack, state reshape)
  into the graph's static buffers.
- Try three flow steps only if the arm-action error and the ten-episode gate stay comfortably
  inside their limits; otherwise keep the bit-exact four-step graph.
