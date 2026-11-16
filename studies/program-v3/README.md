# Program v3: the size search without our code or the token ids

Paused after one scored node. The run's evidence is not committed: it sits uncommitted in its
worktree, so nothing under this directory is a result yet.

## Question

The v1 and v2 discovery agents could read our compression code and were handed the 53 prompt token
ids in the calibration, and many nodes used both. What does the same search find without either?

## Setup

v3 is program-v2's protocol (checkpoint, suite, seeds, gates, statistics, budgets, open policy
agent, memory rule) with two hints removed. It starts from the default policy with no history; no
v1 or v2 node is pooled with it. The choices are recorded in [DREAM_RSI.md](../../docs/DREAM_RSI.md),
section 5.

## Result so far

One node scored: smoke 8/10 at a size reduction of 0.28 (4.01 of 5.60 GB resident). The second
node was running when the run was paused. No cycle completed, no node was promoted, and no reserved
trial exists. Nothing is comparable with v2 yet.

## Evidence

The run's records (`cycles/`, `policies/`, `loop_state.json`, `protocol.json`, `calibration.json`)
live in its worktree and in the Git tag `program-v3-debiased`. They are copied here, with relative
paths, when the run ends.

## Reproduction

```sh
uv run --no-sync vla-compress derive-calibration --out outputs/compression/calibration-v3.pt
uv run --no-sync vla-rsi start --study-dir studies/program-v3 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running `vla-rsi start` resumes from `loop_state.json` and reruns the node that was in
progress. `vla-rsi status`, `vla-rsi stop` and `vla-rsi watch` take the same `--study-dir`.
