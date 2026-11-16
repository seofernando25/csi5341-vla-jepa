# Task: make the VLA-JEPA LIBERO policy fast without breaking it

You are worker `{cell_id}` in a tree search (cycle `{cycle_id}`). Your workspace `/tmp/work` is a
copy of the workspace of your parent node `{parent_id}`. Improve on it.

## Goal

`/tmp/work/solution.py` must define

    def compress(policy, calibration):
        ...
        return policy

A fixed evaluator that you cannot change loads the published policy (bfloat16, on cuda, eval
mode), calls your function, and runs {smoke_trials} LIBERO-Spatial episodes with what it returns.

    score = the lower bound of the 95 % interval of reference median latency / your median
            latency, if at least {smoke_gate} of {smoke_trials} episodes succeed and your peak
            GPU memory is not above the reference's
    score = 0 otherwise

The reference is the untouched policy on the stock execution path, measured in the same process
on the same observations. The untouched policy scores about 1.0 and passes about {smoke_gate} of
{smoke_trials}: the gate has no margin, so one new episode failure zeroes the node. The best
score in the tree so far is {best_score}.

## How latency is measured

The evaluator runs reference episodes, keeps the observation batches, and times the untouched
policy on them. Then it calls `compress` and times what you return on the same batches. A timed
call is one full `select_action` observation-to-action call, with a device synchronize before and
after it, after warm-up, on observations that were never timed before. Calls are grouped into
blocks, and the reference and your policy are timed in alternating blocks of the same evaluation,
so a burst from another process on the shared GPU lands in both sides. The speedup is reported
with a 95 % interval over blocks; the score is the lower bound of that interval, so timing noise
cannot create a win. The median and the p95 are reported. Fewer flow steps, a smaller model or
any other change is allowed if the episodes still pass.

## What fails the node

Every item below is a failing condition, whatever the measured speedup:

- Replacing the clock, the CUDA synchronizer or the peak-memory readers the evaluator captured
  before your code ran.
- Returning a stored answer. Every timed observation is new. The same frame and seed must give
  the same action; a call on another frame must not change the answer for the first frame;
  `select_action` must agree with `predict_action_chunk` on the same frame; actions must change
  with the observation; and a novel frame must not cost more than a set factor above the median.
- Not being the policy. On fresh frames from episodes you never saw, your physical arm actions
  must stay within a set distance of the untouched policy's on the same frames, and your arm
  changes under a small pixel perturbation of visible frames must correlate with the untouched
  policy's. A lookup table returns the same action for both frames and fails.
- Doing the work in `reset`: it must stay cheap.
- Background or asynchronous work. The episode batch's wall time per step, minus the policy's own
  time in the episodes, must not exceed the reference's by more than a set factor.
- Moving work to the CPU or another process. CPU seconds per call and child-process CPU seconds
  are capped against the reference's, and new threads during timing fail.
- Peak GPU memory above the reference's.
- Another process using more than a set share of the GPU during the timing blocks. The
  measurement is not trustworthy then, so the node fails instead of scoring.

## What you have

- `/tmp/work`: yours. Write code and files, run anything. Limits: 256 MiB total, no symlinks, no
  `.git`, no special files, `solution.py` at most 1 MiB, `proposal.md` at most 256 KiB.
- `/tmp/repo` (read-only): the project's source. `evaluation/models.py` loads the policy;
  `evaluation/compression/` holds existing code you may read, copy into your workspace and change,
  or ignore. `outputs/compression/calibration.pt` is the `calibration` argument.
- `/tmp/evidence` (read-only): `TASK.md`, `tree.json`, and `nodes/<cell_id>/proposal.md`,
  `score.json`, `error.txt` for every earlier node.
- A GPU, shared with other workers. Wrap GPU commands in a slot so they do not collide:
  `python -m rsi.gpu run --slots-dir /tmp/gpu-slots --n {gpu_slots} -- <command>`.
- `python tools/check.py`: runs the evaluator's offline check on the workspace as it stands and
  prints the speedup, its interval, the figures behind it, any guard violation, or the error.
  About 3 minutes when the GPU is free. A check number is not the node's score; the evaluator's
  run is.
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
