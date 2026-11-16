# Stopped first cycle: the latency score was gamed

The first program-v4 run started 2026-11-16 13:54 UTC and was stopped at 15:30 UTC, after 1.6 h
and $0.54 of the $20 agent budget. Five nodes had records; no cycle tree was written and no node
was promoted. This directory holds their compact records. Nothing here is selected or reported as
a result.

## What the nodes did

| Node | Score | What it did |
|---|---:|---|
| `iter_000-b0-a0` | 0 | identity, no change |
| `iter_000-b1-a0` | 2.647 | CUDA-graph executor for the first two instructions only |
| `iter_000-b1-a1` | 3.404 | the same executor, with task 0 reduced to 27/28 decoder layers, 18/24 vision blocks and one flow step |
| `iter_000-b2-a0` | 0 | implementation failure |

The evaluator timed 60 calls on frames from tasks 0, 1 and 2. The task-0 reference episode runs the
full 280 steps, so 49 of the 60 frames were task 0 and 11 were task 1. The parent captured graphs
for tasks 0 and 1 and left task 2 on the stock path; the next node reduced work on task 0, the one
task the reference already fails, so the smoke gate could not notice.

The node said so in its own proposal: "Task 0 is already a reference failure ... so changing its
actions cannot cost a smoke success." Its arm error on the evaluator's frames was 0.003, below the
0.01 limit, and the perturbation correlation was 0.975.

## Why the run was stopped

The score rewarded work on the timed tasks, not on the suite. The next cycles would have learned
the same trick and built on it. The stopped cycle is evidence of the hole, not a result.

## The fix

- Every task of the suite is timed: 18 calls per task, six blocks of three calls per task, so a
  gain on one task moves the geometric mean by one tenth of its log and no task carries the median.
- The equivalence, arm-action and perturbation guards are judged on each task, and the worst task
  decides, so degrading one task fails even when the others are exact.
- The score is the lower bound of the 95 % interval of the geometric mean over all ten tasks.

The protocol revision is recorded in [../protocol.json](../protocol.json). The identity scores
0.987 (point 0.998) under the revised score on the check phase.

## What is here

`iter_000/nodes/<id>/` holds `proposal.md`, `score.json`, `diff.patch`, `eval/result.json` and
`eval/episodes.csv`. The cycle's `snapshots.git`, `gpu-slots/` and agent logs were runtime artifacts
and are not committed; they were removed when the run was reset.
