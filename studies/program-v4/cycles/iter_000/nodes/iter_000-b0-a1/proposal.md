# Proposal: bit-exact CUDA-graph execution without the per-call eval walk

## Change relative to the parent

The parent (`iter_000-b0-a0`, score 2.706) routes `predict_action_chunk` through a
bit-exact CUDA-graph executor (`graphed.py`) and drops the training-only V-JEPA branch.
This node keeps that executor and makes three further changes, all of them bit-exact on
every observation-bank frame:

1. **A lean `select_action`.** The stock `VLAJEPAPolicy.select_action` calls `self.eval()`
   before every prediction. `nn.Module.eval()` walks every module of the 2B backbone and
   measured **1.48 ms** per call on this GPU; the evaluator loads the policy in eval mode
   and never trains it, so the walk is pure overhead. `populate_queues` only fills the
   observation queues, which are unused with `n_obs_steps == 1` because
   `_prepare_model_inputs` reads the batch directly. The replacement keeps the action
   queue exactly as the stock method keeps it (extend with
   `actions.transpose(0, 1)[:n_action_steps]`, then `popleft`), so the sequence of actions
   over an episode is unchanged. This is the whole gain: `select_action` fell from
   27.96 ms to 26.42 ms in the development profile, while the graph replay itself is
   25.8 ms.
2. **The decoder rotary table is precomputed per prompt length** instead of being
   recomputed inside every replay. It depends only on the prompt's position ids and the
   hidden dtype, so the values are identical; it is stored in the graph plan.
3. **Images are copied into the static buffer view by view** instead of allocating a
   stacked copy first. Same values, one less 1.2 MB allocation and copy per call.

The graph path is still bit-exact with the untouched policy: the check reports arm action
error 0 on the worst task and perturbation correlation 1.000, and the action spread /
reproducibility / `select_action`-vs-`predict_action_chunk` guards pass. The graph is
captured in `compress` for the ten suite instructions, so no capture cost lands in a
timed call.

## Why it should be faster

The profile of the parent's timed path was: graph replay 25.8 ms, `select_action` 28.0 ms.
The 2.2 ms difference was almost entirely the redundant `self.eval()` tree walk and the
input copies. Removing the walk and the extra stack copy moves the candidate median from
28.7 ms to 26.5 ms with no numeric change, so the speedup rises from about 2.7x to about
2.85x. Because the change is bit-exact, the 8/10 episode gate is reproduced exactly (the
reference fails tasks 0 and 8; the candidate fails the same task 0 episode in the check).

## Last check output

`PYTHONPATH=/tmp/repo /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 15.5% (resident 5215956992 of baseline 6169967616 bytes)
score: 2.8312 (latency_speedup)
speedup: point 2.8517, 95% [2.8312, 2.9207], 6+6 blocks, interleaved
speedup per task: 0 2.845, 1 2.849, 2 2.880, 3 2.894, 4 2.867, 5 2.837, 6 2.834, 7 2.826, 8 2.853, 9 2.834
latency ms: reference median 74.81, candidate median 26.52, p95 30.16
arm action error (worst task): 0 (limit 0.01)
perturbation correlation (worst task): 1.000 (min 0.5)
other GPU use: max 0.0 %, 12 samples
```

The parent's check on the same harness was 2.6718 (candidate median 28.79 ms), so this is
a ~6 % latency gain with identical actions. The check's task-0 episode fails exactly as
the reference's task-0 episode does; the reference also fails task 8, and the bit-exact
path therefore reproduces the reference's 8/10 development gate.

## What I would try next

- Cache the flow head's cross-attention K/V projections (the conditioning tokens are
  constant across the four flow steps) and the state-encoder output; both are bit-exact
  and should remove ~50 small GEMMs per call.
- Move the eager 256→224 area resize into the graph so the static buffer holds the raw
  observation; it is bit-exact and saves the remaining ~0.1 ms of input preparation.
- The graph replay is still 25.8 ms with ~4600 kernels, so kernel-count reduction is the
  remaining lever. `torch.compile` cut it to 20.5 ms in node `iter_000-b2-a0` but that
  node failed the smoke gate (7/10) from the resulting bf16 drift, so a compiled variant
  would need a full ten-episode gate validation (and a deterministic kernel selection)
  before it is safe to submit. Compiling only the flow head or only the vision tower
  would produce a smaller drift and is the next thing to test.
