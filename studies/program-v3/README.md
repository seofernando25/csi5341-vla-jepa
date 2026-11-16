# Program v3: the same search without the two hints

Started 2026-11-16. A fresh Dream-RSI run from the baseline policy with no history, under
program-v2's protocol with two hints removed from what the discovery agents can see. The frozen
rules are in [protocol.json](protocol.json); `supersedes` there states what this namespace
replaces.

## Question

How much of v2's compression came from the two hints we handed the discovery agents: the
project's own compression code at `/tmp/repo/evaluation/compression`, and the 53 prompt token ids
in the calibration? Without them, what does the same search find at the same gates?

## Scope

- Model: `lerobot/VLA-JEPA-LIBERO` at `735d9f692981e286ade093b5046627eda876e5d0`.
- Suite: LIBERO-Spatial, tasks 0-9. Same seeds, phases, gates, statistics and budgets as
  program-v1 and program-v2.
- The search starts with no trees, no beta sweeps, no scores and the default policy
  (`rsi/policies/optimal.py`). No v1 or v2 node is pooled with a v3 node or used as its starting
  point.
- Reserved-500 states are never used for selection; the lock is on.
- Calibration: `outputs/compression/calibration-v3.pt` (runtime, not in Git), derived from the
  frozen program-v1 artifact with `token_ids` removed. [calibration.json](calibration.json) is
  its frozen description.

## Difference from program-v2

Everything else is v2's protocol: open policy agent, live `plan_grid` under caps, the fixed
memory rule, the same gates, seeds, phase sizes, agent, timeouts and budgets.

| Field | program-v2 | program-v3 |
|---|---|---|
| Discovery agent's `/tmp/repo` | full repo source, `evaluation/compression/` readable and copyable | `evaluation/compression`, `evaluation/nf4.py`, `evaluation/fastpath.py` and `outputs/compression/vendor` not mounted; `evaluation/models.py`, `evaluation/pipeline.py` and the model source stay readable; installed packages stay available |
| Calibration handed to `compress` | `linear_inputs`, `block_influence`, `token_ids`, `bank_indices` | `linear_inputs`, `block_influence`, `bank_indices`; no `token_ids`, no per-prompt information |
| Workspace `tools/check.py` | runs the fixed evaluator out of `/tmp/repo` | self-contained: the evaluator's resident walk and payload sweep copied into `tools/`, plus the frozen reference actions; no `evaluation.compression` import |
| Prompt and workspace README | name `evaluation/compression/` and `token_ids` | name neither |

Why the check differs: the check tool used to run the evaluator inside the agent's sandbox, and
the evaluator imports `evaluation/compression`. With that path unmounted, the check measures the
same quantities with the evaluator's own primitives (`evaluation/compression/resident.py`,
`memory.py`) copied into `tools/` at template time. The evaluator keeps its own sandbox with the
full source; its numbers still decide the score.

## Reproduction

From a checkout of the run branch, with `outputs/compression/` and `outputs/evaluation/`
populated as `rsi/sandbox.py` binds them:

```sh
uv run --no-sync vla-compress derive-calibration --out outputs/compression/calibration-v3.pt
uv run --no-sync vla-rsi start --study-dir studies/program-v3 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running the same command resumes from `loop_state.json`. `vla-rsi status`, `vla-rsi stop` and
`vla-rsi watch --port 8788` take the same `--study-dir`. The pre-GPU check is the test suite; a
synthetic calibration run is a harness check, not evidence.

## Results

Written when the run stops. `vla-rsi status --study-dir studies/program-v3` reports progress.

## What the comparison can and cannot say

v3 and v2 share the gates, the seed, the evaluator and the memory rule, so a v3 node is
comparable to a v2 node at the same phase. The comparison is not a clean ablation of one hint:
the mount and the check tool changed together, and a v3 agent that wants a feedback loop must use
the self-contained check. A v3 result below v2's is evidence that the hints helped; a v3 result
at or above v2's is evidence the search did not need them. Neither says what an agent given the
code but not the token ids would find.
