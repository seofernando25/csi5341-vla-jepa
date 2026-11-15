# Task: make the VLA-JEPA LIBERO policy smaller without breaking it

You are worker `{cell_id}` in a tree search (cycle `{cycle_id}`). Your workspace `/tmp/work` is a
copy of the workspace of your parent node `{parent_id}`. Improve on it.

## Goal

`/tmp/work/solution.py` must define

    def compress(policy, calibration):
        ...
        return policy

A fixed evaluator that you cannot change loads the published policy (bfloat16, on cuda, eval
mode), calls your function, measures every byte of tensor memory reachable from what you return,
and runs {smoke_trials} LIBERO-Spatial episodes with it.

    score = fraction of memory removed, if at least {smoke_gate} of {smoke_trials} episodes succeed
    score = 0 otherwise

The uncompressed policy scores 0. The best score in the tree so far is {best_score}.

## What you have

- `/tmp/work`: yours. Write code, add files and packages, run anything. `solution.py` may import
  from the workspace and from anything installed.
- `/tmp/repo` (read-only): the project's source. `evaluation/models.py` loads the policy;
  `evaluation/compression/` holds existing quantization, pruning and calibration code that you may
  import, copy into your workspace and change, or ignore. `outputs/compression/calibration.pt` is
  the `calibration` argument.
- `/tmp/evidence` (read-only): `tree.json` and, for every earlier node, `proposal.md`,
  `score.json` and `error.txt`.
- A GPU, shared with other workers. Wrap GPU commands in a slot so they do not collide:
  `python -m rsi.gpu run --slots-dir /tmp/gpu-slots --n {gpu_slots} -- <command>`.
- `python tools/check.py`: runs the evaluator's offline check on the workspace as it stands and
  prints memory removed, action error against the uncompressed policy on 80 held-out frames,
  latency, or the error. About 20 s when the GPU is free. Action error is not task success, but a
  large error predicts failure.
- No internet packages. {minutes} minutes of wall time; a worker that runs over is killed and
  its node fails.

## Rules

- Memory is counted wherever it hides: tensors on the CPU, in lists, in module globals, in files
  reloaded later. Hiding memory makes the node fail. Really remove it.
- Do not edit `tools/`.
- A workspace identical to an existing node is rejected.

## Deliverables

A node has exactly two required outputs in `/tmp/work`:

- `solution.py`: must define the top-level `compress(policy, calibration)` shown above. The
  evaluator scores this file.
- `proposal.md`: what you changed relative to the parent, why, the last check output, and what you
  would try next.

Both files must exist and be readable when you stop. A missing or unreadable `proposal.md` fails
the node before evaluation, whatever `solution.py` does; so does a `solution.py` that does not
parse or defines no top-level `compress`. Write `proposal.md` early and update it as you go.

## How to work

1. Read `/tmp/evidence` first: what was tried, what scored, what failed and why.
2. Pick one idea that differs from your parent and from failed siblings. Any method is allowed:
   quantization, pruning, low-rank factors, weight sharing, removing unused parts, a different
   numeric format, something not listed here.
3. Implement it in the workspace. Run `python tools/check.py`. Fix errors. Iterate while time
   allows.
4. Write `/tmp/work/proposal.md`: what you changed relative to the parent, why, the last check
   output, and what you would try next. Keep it current as you iterate.

Your node is scored on `solution.py` as it is when you stop. `proposal.md` is required: a node
without it fails.
