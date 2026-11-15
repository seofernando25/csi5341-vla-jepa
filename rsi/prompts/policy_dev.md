# Task: improve an exploration policy (step {step} of {steps}, cycle {cycle_id})

You improve the search policy for a discovery tree. Each node is a candidate model-compression
recipe written by another agent. A node's score is the fraction of model size it removed if it
passed its quality gate, else 0. The policy only decides which nodes to expand next and when to
stop. You do not design recipes.

## Files

- `/tmp/work/method.py`: the policy. The only file you may change. It starts as the best policy
  found so far.
- `/tmp/evidence/worlds/*.json`: recorded trees the policy is replayed on.
- `/tmp/evidence/beta_sweep.json`, `/tmp/evidence/policy_execution_traces.jsonl`: results and
  per-round decisions of the policy in `method.py` as it stands now.
- `/tmp/evidence/history/step_NN/`: every earlier step of this cycle, with its `method.py`,
  `scaling_curve.json` (probes, attainment and rounds per world and beta, and V) and
  `verdict.json` (accepted or rejected, and why).
- `/tmp/evidence/live_cycle_manifests.json`: summaries of past live cycles.
- `/tmp/evidence/reference/`: read-only copies of `api.py`, `tree.py`, `objective.py`.

## Objective

The policy is replayed on every world at beta in (0, 0.25, 0.5, 0.75, 1). Each replay gives
probes used `n` and attainment `a` = best score found / best score in the tree. Per world,

    V = area under the best quality-versus-cost frontier of those five points
        - {lambda_parallel} * mean(sequential rounds / probes)

A batch of up to {max_parallelism} probes costs one round. V is high when the policy reaches the
best node with few probes, stops when nothing more can be gained, and fills its batches. Your
policy replaces the current one only if its mean V over all worlds is strictly higher.

## Rules (machine-checked; a violation rejects the step)

- Keep `class OptimalPolicy(LLMDesignedMethod)` with `NAME = "OptimalPolicy"`.
- Imports: `from __future__ import annotations`, `import math`, `from rsi.api import ...`,
  `from dataclasses import dataclass, field`, `from typing import ...`. Nothing else.
- `solve(self, question, budget=None)` may read only `question.observed()`,
  `question.legal_actions()`, `question.legal_roots()`, `question.opened_branches()`,
  `question.meta(cell_id)`, `question.baseline_score`, `question.max_parallelism`, and may call
  `question.probe_batch(batch)`. Count your own probes; do not read `question.budget_spent`,
  `question.best_so_far` or any `_private` attribute of another object.
- `beta = float(self.config.get("beta", default))` is read once in `solve()`. Every threshold
  comes from `_schedule(beta)`. Changing only the default beta changes nothing: all five betas
  are replayed anyway.
- Deterministic: no randomness, clocks, files, or global state.
- `solve()` terminates when the budget is spent or no legal action remains.
- Do not change `plan_grid`.
- No string or number in the file may be copied from a recorded world (cell ids, world ids,
  exact scores, node counts). Such a file is rejected.

## How to work

1. Read the scaling curve and the traces. Find where the current policy wastes probes, stops too
   early, stops too late, or sends batches smaller than {max_parallelism}. Name the world, beta
   and round.
2. State one hypothesis as a rule about tree structure that would hold in a tree you have not
   seen, for example "children of a root that failed at runtime rarely pass". Check it against
   every recorded world, not only the one that suggested it.
3. Make the smallest change to `_schedule` or `solve` that implements the rule.
4. Read `history/`. Do not repeat a change that lowered V or was rejected.
5. You cannot run code. The harness replays your file after you finish; its result is shown to
   the next step. The policy is also replayed on hidden trees of other shapes, and that result is
   reported next to V.

End with three lines: the hypothesis, the change, the expected effect on V.
