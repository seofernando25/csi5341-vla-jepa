# Program v3: the size search without our code or the token ids

Three cycles of six probes. The best node of each cycle that passed development ran the reserved
500 once.

## Question

The v1 and v2 discovery agents could read our compression code and were handed the 53 prompt token
ids in the calibration, and many nodes used both. What does the same search find without either?

## Setup

v3 is program-v2's protocol (checkpoint, suite, seeds, gates, statistics, budgets, open policy
agent, memory rule) with two hints removed ([protocol.json](protocol.json)). It starts from the
default policy with no history; no v1 or v2 node is pooled with it. The choices are recorded in
[DREAM_RSI.md](../../docs/DREAM_RSI.md), section 5.

Run history, all 2026-11-16 and 2026-11-17:

- Cycle 1 ran on code revision `cef9f96` and finished with development 82/100 on its best node.
- The run was then stopped: cycle 2 made no probe because the selected policy proposed nothing on
  an empty tree, and the cost guard could not be enforced. The cycle 2 state was reset
  (`iter_001`), the loop was resumed on `d875913`, and cycles 2 and 3 ran there.
- `d875913` adds a fallback: when the selected policy makes no probe on an empty tree, the
  built-in `rsi/policies/optimal.py` drives the cycle. Cycles 2 and 3 both used it
  (`policy_fallback` in their manifests), so the policy agent's output did not choose their probes.
  The policy agent did select a policy after each cycle ([policies/](policies)); held-out value
  was below the incumbent's each time.
- `iter_001-b0-a0` failed with `EFBIG`: the agent's own test runs flooded stdout past the
  sandbox's 64 MiB file cap. It is a harness failure, scored 0, and counts as a probe.
- The policy-development sandbox could list `evaluation/compression` and the vendor tree. The
  logs show `find` listings, and no tool call read either.
- The prompt token ids are not read from the calibration, but several nodes still hardcode them
  (collected by running the policy's own processor) or tokenize the ten task instructions at
  load time. Cycle 1's best node does the first, the others the second.

## Result

Selection rule, fixed from the cycle records before any reserved run
([selection.json](selection.json)): each cycle's best node that passed development (77/100). The
brief named two candidates; the records show a third, the cycle 2 best, which also passes.

| Candidate | Memory removed | Dev | Reserved | Paired vs B16 |
|---|---|---|---|---|
| iter_000-b2-a2 | 81.9 % | 82/100 | 471/500 | -0.8 [-2.6, 1.2] |
| iter_001-b2-a3 | 80.0 % | 83/100 | 477/500 | +0.4 [-1.2, 2.2] |
| iter_002-b2-a2 | 82.8 % | 83/100 | 480/500 | +1.0 [-0.6, 2.6] |

B16 scored 475/500 on the same states. The pair difference is in points, task-stratified, 95 %
bootstrap interval. No candidate differs from B16 in success; the best removes 82.8 % of resident
memory (0.965 GB of 5.60 GB), about the same as v1 and v2. Tasks 0, 5 and 8 lose the most in
development (3 to 5 of 10); the smoke gate, which fails only tasks 0 and 8, did not predict that.

`iter_002-b2-a2`'s first reserved launch was killed by the tooling after 123 episodes, with no
result read; it was rerun once from the start ([note](reserved/iter_002-b2-a2/README.md)).

## Evidence

- [selection.json](selection.json), [reserved/summary.json](reserved/summary.json): the frozen
  selection and the reserved counts, Wilson intervals and paired differences, from
  [reserved/summarize.py](reserved/summarize.py).
- `reserved/<cell>/final/`: the reserved episodes and evaluator record of each candidate.
- `cycles/iter_00N/`: `tree.json`, `grid_plan.json`, `live_cycle_manifest.json`, and per node the
  proposal, score, diff against its parent and, for promoted nodes, the development record.
- `policies/iter_00N/selection.json`: the policy agent's selection after each cycle. The policy
  files and loop state are not committed.

## Reproduction

```sh
uv run --no-sync vla-compress derive-calibration --out outputs/compression/calibration-v3.pt
uv run --no-sync vla-rsi start --study-dir studies/program-v3 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

The reserved runs export a node's snapshot and run the fixed evaluator once:

```sh
uv run --no-sync vla-compress solution --workspace <exported node> --phase final \
  --allow-reserved --protocol studies/program-v3/protocol.json \
  --out studies/program-v3/reserved/<cell>/final
uv run --no-sync python studies/program-v3/reserved/summarize.py
```

Set `TMPDIR` and `TRITON_CACHE_DIR` to the same scratch directory so compile caches stay inside the
evaluator's allowed paths. The Git tag `program-v3-debiased` holds the run's code and snapshots.
