# Proposal: CUDA-graph execution of the VLA-JEPA action chunk

## Change relative to the parent

The parent is the identity policy (`compress` returns the loaded B16 policy).  This node
routes `policy.predict_action_chunk` (and therefore `select_action`) through a CUDA-graph
executor adapted from the project's `evaluation/fastpath.py`:

- The same vision tower, Qwen decoder, DeepStack injections and four-step flow head are
  composed without the stock path's host synchronizations (vision `.tolist()` and the
  decoder mask code).
- One graph is captured per distinct prompt length.  The ten LIBERO-Spatial instructions
  have five distinct token lengths, so five graphs share one graph pool.
- All graphs share the static image, state, noise and output buffers and the vision
  position/rotary tables.  Text embeddings and the decoder rotary table are computed
  inside the graph from per-instruction token ids, so no per-task language prompt copy is
  stored.
- `compress` also drops the V-JEPA video encoder, predictor and processor.  They are a
  training-only branch and inference never calls them; dropping them frees their weights
  and gives the graph pool a wide peak-memory margin.
- `compress` captures the graphs for the ten suite instructions before the evaluator's
  warm-up and timed blocks.  An unseen instruction falls back to the same lazy graph
  capture, so the policy still works.

The graph path was compared against the untouched policy on the observation bank for all
ten tasks: maximum absolute action difference was 0.0 and `torch.allclose` passed.  The
graph path preserves the stock noise draw (`torch.randn` with the same shape, dtype and
device before replay), so `select_action`/`predict_action_chunk` agreement,
reproducibility and perturbation response are unchanged.

## Why it should be faster

The stock path is launch-bound: each prediction launches thousands of small kernels and
synchronizes with the host inside the vision tower and decoder.  Replaying one captured
graph removes launch overhead and host synchronization.

## Last check output

`PYTHONPATH=/tmp/repo /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 15.5% (resident 5216197120 of baseline 6169967616 bytes)
score: 2.6718 (latency_speedup)
speedup: point 2.6935, 95% [2.6718, 2.7233], 6+6 blocks, interleaved
speedup per task: 0 2.729, 1 2.732, 2 2.671, 3 2.711, 4 2.674, 5 2.671, 6 2.707, 7 2.668, 8 2.703, 9 2.669
latency ms: reference median 77.32, candidate median 28.79, p95 29.96
arm action error (worst task): 0 (limit 0.01)
perturbation correlation (worst task): 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The check's `result.json` recorded a candidate peak CUDA of 5.216 GB against a reference
peak of 6.170 GB, so the peak-memory gate has a wide margin.  The candidate episode on
task 0 failed exactly as the reference episode on task 0 did; all other reference episodes
succeeded.

## What I would try next

- If the final evaluator's peak figure is unexpectedly high, reduce the number of captured
  graphs further (for example keep only the most common prompt lengths and let the others
  use the stock path) or shrink the shared vision buffers.
- If an episode failure appears, compare the graphed and stock rollouts on the failing
  task and seed; the graph path is bit-exact on bank frames, so a failure would point to
  an episode-specific input shape or processor state.
- If more speed is needed, the remaining overhead is the stock `_prepare_model_inputs`
  image resize and the per-call input copies; moving the resize into the graph would need
  a matching resampling kernel to stay bit-exact and was not attempted here.
