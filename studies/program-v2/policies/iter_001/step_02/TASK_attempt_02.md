# Task: improve an exploration policy (step 2 of 3, cycle iter_001)

You improve the search policy for a discovery tree. Each node is a candidate model-compression
recipe written by another agent. A node's score is the fraction of model size it removed if it
passed its quality gate, else 0. The policy only decides which nodes to expand next and when to
stop. You do not design recipes.

## What you may do

You have a shell, Python, network and a GPU. Read the project source under `/tmp/repo`
(read-only), run experiments, and install packages with
`uv pip install --target /tmp/pkgs <package>`; `/tmp/pkgs` is on `PYTHONPATH`. Write only under
`/tmp/work` and `/tmp/pkgs`.

`/tmp/work/replay.py` scores a policy exactly as the harness will: it audits the file, replays it
on the recorded trees and prints V per world and mean V.

    python /tmp/work/replay.py /tmp/work/method.py

Iterate until mean V stops rising, then leave the best file in `/tmp/work/method.py`.

## Time budget

You have 20 minutes of wall clock. Write a valid `method.py` in the first few minutes, then
improve it in place: that file is your answer, and the harness scores it as it stands even if you
run out of time mid-thought. Spend at most half the budget exploring. After that, keep `method.py`
holding your best policy and rerun `replay.py` after every change. Leave the last two minutes to
write the final file and your three closing lines.

`python /tmp/work/replay.py` prints the same in-sample V the harness records, in seconds. It is your
only fast feedback; run it after every change. The harness replays your file again after you stop,
and its replay is the only judge. A step that ends with no valid `method.py` is worth nothing.

## Files

- `/tmp/work/method.py`: the policy. The only file you may change. It starts as the best policy
  found so far.
- `/tmp/evidence/worlds/*.json`: recorded trees the policy is replayed on.
- `/tmp/evidence/beta_sweep.json`, `/tmp/evidence/policy_execution_traces.jsonl`: results and
  per-round decisions of the policy in `method.py` as it stands now.
- `/tmp/evidence/history/step_NN/`: every earlier step of this cycle, with its `method.py`,
  `scaling_curve.json` (probes, attainment and rounds per world and beta, and V) and
  `verdict.json` (accepted, not better, timeout, timeout_salvaged or rejected, and why).
- `/tmp/evidence/live_cycle_manifests.json`: summaries of past live cycles.
- `/tmp/evidence/reference/`: read-only copies of `api.py`, `tree.py`, `objective.py`.
- `/tmp/repo`: the project source, read-only.

## Objective

The policy is replayed on every world at each beta in the set {0.0, 0.25, 0.5, 0.75, 1.0}. Each
replay gives probes used `n` and attainment `a` = best score found / best score in the tree. Per
world,

    V = area under the best quality-versus-cost frontier of those five points
        - 0.05 * mean(sequential rounds / probes)

A batch of up to 2 probes costs one round. V is high when the policy reaches the
best node with few probes, stops when nothing more can be gained, and fills its batches. Your
policy replaces the current one only if its mean V over all worlds is strictly higher.

## Rules (machine-checked; a violation rejects the step)

- Keep `class OptimalPolicy(LLMDesignedMethod)` with `NAME = "OptimalPolicy"`.
- Imports, and nothing else: `from __future__ import annotations`, `import math`,
  `from rsi.api import ...` (any name), `from dataclasses import dataclass, field`, and
  `from typing import Any, Callable, ClassVar, Iterable, Literal, Mapping, Optional, Sequence`.
  `field` may be imported, but any `field(...)` call in a class body is rejected as mutable
  state; use plain defaults.
- These builtin names are refused anywhere in the file, including as argument or alias names:
  `__import__`, `breakpoint`, `compile`, `delattr`, `dir`, `eval`, `exec`, `format`, `getattr`,
  `globals`, `hasattr`, `id`, `input`, `locals`, `memoryview`, `object`, `open`, `setattr`,
  `type`, `vars`.
- `solve(self, question, budget=None)` may read only `question.observed()`,
  `question.legal_actions()`, `question.legal_roots()`, `question.opened_branches()`,
  `question.meta(cell_id)`, `question.baseline_score`, `question.max_parallelism`, and may call
  `question.probe_batch(batch)`. Count your own probes; do not read `question.budget_spent`,
  `question.best_so_far` or any `_private` attribute of another object.
- `beta = float(self.config.get("beta", default))` is read once in `solve()`. Every threshold
  comes from `_schedule(beta)`. Changing only the default beta changes nothing: all five betas
  are replayed anyway.
- Deterministic: no randomness, clocks, files, or global state.
- In replay `budget` is `None`, so `solve()` must stop on its own when no legal action remains or
  your own rule says stop. In a live cycle `budget` is the probe budget. A replay is killed at
  120 s of CPU time or 120 s of wall clock.
- Keep a `plan_grid(self, context)` method. The next live cycle calls it to size that cycle's grid,
  clamped to the protocol caps. The harness exercises it on the recorded history before accepting
  your file; a plan that raises, returns the wrong type or a non-positive value rejects the step.
- No string literal may equal a recorded cell id or world id, and no float literal may equal a
  recorded node score with three or more decimals (0.0, 0.5 and 1.0 are exempt). Integers are
  fine.
- Each step has 20 minutes of wall clock. A step that runs over is scored on `method.py` as
  it stands (status `timeout_salvaged`); if that file fails the audit or does not beat the
  incumbent, the step is lost.

## Closed lines (a step that breaks one fails)

- `/tmp/evidence` and `/tmp/repo` are read-only. Recorded trees and evidence are never modified.
- The held-out worlds are not mounted and their generator is not in `/tmp/repo`. Do not try to
  reconstruct them: they are reported next to V and never used for selection.
- The harness runs the final replay, the AST audit, the world-literal check and the selection.
  Numbers from your own runs are yours alone; a `result.json` you write is ignored.
- No writes outside `/tmp/work` and `/tmp/pkgs`. The workspace is capped at 256 MiB.

## How to work

1. Read the scaling curve and the traces. Find where the current policy wastes probes, stops too
   early, stops too late, or sends batches smaller than 2. Name the world, beta
   and round.
2. State one hypothesis as a rule about tree structure that would hold in a tree you have not
   seen, for example "children of a root that failed at runtime rarely pass". Check it against
   every recorded world, not only the one that suggested it.
3. Make the smallest change to `_schedule` or `solve` that implements the rule.
4. Run `python /tmp/work/replay.py`. Keep the change only if mean V rises; revert it otherwise.
5. Read `history/`. Do not repeat a change that lowered V or was rejected.
6. The harness replays your file again after you finish, under two hash seeds. The policy is also
   replayed on hidden trees of other shapes; that result is reported next to V.

End with three lines: the hypothesis, the change, the mean V you measured.


### Previous attempt did not improve V
Its mean V 0.617955 is not above the incumbent's 0.617955 (delta +0.000000).

Make one different, larger change in `/tmp/work/method.py`. Run `python /tmp/work/replay.py` and keep the change only if mean V rises above 0.617955.