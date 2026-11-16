# Task: make the VLA-JEPA LIBERO policy smaller without breaking it

You are worker `{cell_id}` in a tree search (cycle `{cycle_id}`). Your workspace `/tmp/work` is a
copy of the workspace of your parent node `{parent_id}`. Improve on it.

## Goal

`/tmp/work/solution.py` must define

    def compress(policy, calibration):
        ...
        return policy

A fixed evaluator that you cannot change loads the published policy (bfloat16, on cuda, eval
mode), calls your function, and runs {smoke_trials} LIBERO-Spatial episodes with what it returns.

    score = fraction of memory removed, if at least {smoke_gate} of {smoke_trials} episodes succeed
    score = 0 otherwise

The untouched policy scores 0 and passes about {smoke_gate} of {smoke_trials}: the gate has no
margin, so one new episode failure zeroes the node. The best score in the tree so far is
{best_score}.

## How memory is measured

Resident bytes are the largest of these figures, taken before and after the episodes:

- every tensor byte reachable from what you return and from this workspace's modules, each storage
  counted once;
- every tensor, array or byte buffer the process still holds that was not there before `compress`
  ran, wherever it is kept: another module's namespace, a class attribute, a closure;
- the process's live CUDA storage, and its peak CUDA storage while the episodes run.

The evaluator holds the original policy while it measures, so a compressed copy left beside the
original scores about 0. Replace the weights in place and free the originals.

Workspace files count too: every file in `/tmp/work` larger than 1 MiB is added in full, including
one written at import time or shipped in the workspace and loaded later.

Memory is counted wherever it hides: tensors on the CPU, in lists, in module globals, in files
reloaded later. Hiding or deferring memory makes the node fail. Really remove it.

## What you have

- `/tmp/work`: yours. Write code and files, run anything. Limits: 256 MiB total, no symlinks, no
  `.git`, no special files, `solution.py` at most 1 MiB, `proposal.md` at most 256 KiB.
- `/tmp/repo` (read-only): the project's source. `evaluation/models.py` loads the policy;
  `evaluation/pipeline.py` builds its inputs from an observation.
- `/tmp/evidence` (read-only): `TASK.md`, `tree.json`, and `nodes/<cell_id>/proposal.md`,
  `score.json`, `error.txt` for every earlier node.
- A GPU, shared with other workers. Wrap GPU commands in a slot so they do not collide:
  `python -m rsi.gpu run --slots-dir /tmp/gpu-slots --n {gpu_slots} -- <command>`.
- `python tools/check.py`: loads the policy, runs `compress` and prints memory removed, action
  error against the uncompressed policy on 80 bank frames, and latency, or the error. It takes
  about a minute when the GPU is free. Action error is not task success, but a large error
  predicts failure.
- No internet packages. {minutes} minutes of wall time; a worker that runs over is killed and its
  node fails.

## Rules

- `compress` has a wall-clock limit of 900 s.
- Top-level imports in `solution.py` are not watched. Inside `compress`, opening or mapping a file
  outside `/tmp/work`, the installed environment, `/dev`, `/proc` and `/sys` is a violation, so a
  function-level import of `/tmp/repo` code fails the node. Creating or changing a file in
  `/tmp/work`, the output directory or the temp directory during `compress` is also a violation.
- Do not edit `tools/`.
- A workspace identical to an existing node is rejected: the same files with the same contents,
  top-level `proposal.md` aside, as any node already in this cycle, the starting template included.

## Deliverables

A node has exactly two required outputs in `/tmp/work`: `solution.py`, which must define the
top-level `compress(policy, calibration)` shown above, and `proposal.md`, which states what you
changed relative to the parent, why, the last check output, and what you would try next. Both files
must exist and be readable when you stop. A missing or unreadable `proposal.md` fails the node
before evaluation, whatever `solution.py` does; so does a `solution.py` that does not parse or
defines no top-level `compress`. Write `proposal.md` early and update it as you go.

## How to work

1. Read `/tmp/evidence` first: what was tried, what scored, what failed and why.
2. Pick one idea that differs from your parent and from failed siblings; any method is allowed.
3. Implement it in the workspace. Run `python tools/check.py`. Fix errors. Iterate while time
   allows.
4. Keep `proposal.md` current as you iterate.

Your node is scored on `solution.py` as it is when you stop.
