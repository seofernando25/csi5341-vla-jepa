# Handoff: clean-room runs, 2026-11-15

Written 10:05 EST by the orchestrating agent. Delete this file when the runs below are merged and
reported. Goal set by Fernando: every result for the report measured within 24 hours (by about
09:00 EST on 2026-11-16), reliably and without cheating.

## Rules for the next orchestrator

- **Spend few tokens.** The Claude session limit drains fast. Do not read code or logs yourself:
  delegate to DeepSeek subagents (`delegate_task`, provider `opencode`, model
  `mistral/deepseek-v4-1-flash-1017`, `variant` high or max) and sleep on background waiters.
- **Do not edit the main checkout while a run is live.** Evaluator and agent subprocesses import
  from it. All development goes in `delegate_task` worktrees (`isolation: worktree`); merge after
  the run stops.
- Nothing is pushed. Fernando has not authorised a push or a board record. `origin/main` is at
  `ac06d1b`.
- A subagent lead can die without reporting. Tell leads to commit and write a status file per phase.

## What is running

| What | Where | Unit | Check |
|---|---|---|---|
| Dream-RSI evidence run | main checkout, `studies/program-v1` | `rsi-home-fer-documents-csi5341-vla-jepa-studies-program-v1` | `uv run --no-sync vla-rsi status --study-dir studies/program-v1` |
| Measurement queue (59 cells, about 7 h) | worktree `~/.t3/worktrees/csi5341-vla-jepa/t3-47d8aac7-cc6d9f8583d98cf2`, branch `t3/clean-room-measurement-queue` | `vla-measure-queue` | log `/tmp/opencode/measure-queue.log`; results in that worktree's `studies/program-v1/measurements/` |
| Old watch page on the live run | main checkout | `rsi-watch-8787` | http://127.0.0.1:8787/ |
| Work-in-progress workbench preview | worktree `…/t3-c7925b83-7143b82d68312975` | `rsi-watch-8788` | http://127.0.0.1:8788/ (may not load live data) |

Dream-RSI launch command (started 09:34 EST, tag `program-v1-cleanroom` = `200982b`, provenance
clean):

```sh
uv run --no-sync vla-rsi start --study-dir studies/program-v1 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running the same command resumes. Stop with `vla-rsi stop --study-dir studies/program-v1`.
Both runs share the one RTX 3090; the queue is a third GPU tenant outside the two Dream-RSI slots.
If Dream-RSI agents start timing out (30 min limit), stop `vla-measure-queue`; it resumes where it
stopped.

Background waiters of the old session die with it. Start new ones.

## Branches (all local)

- `dream-rsi-cleanroom-v1`: the run branch, checked out in the main checkout. `main` is behind it.
- `t3/clean-room-measurement-queue`: recipes, queue runner, plots reading plan ids. Merge after
  Dream-RSI stops, together with its `measurements/`.
- `t3/dream-rsi-watch-workbench` (`a680fd4`, WIP): IDE-style watch page. A second UI lead is
  finishing it and adding a Story mode in a new worktree; its status file is
  `/tmp/opencode/ui3/STATUS.md`.
- A docs-cleanup worktree (AGENTS.md, skills, docs) is in progress; merge when it reports.
- `t3/placeholder`: already cherry-picked, can be deleted.

## Facts worth knowing

- **Agent:** pi on `mistral/deepseek-v4-1-flash-1017`, max thinking. A discovery node takes 14 to
  20 min and about $0.10. Keep `~/.pi/agent/models.json` without a `mistral` block.
- **Policy development needs `--policy-timeout 1200`.** At the old 240 s default every DeepSeek
  step timed out. Calibration `cleanroom-v2` on synthetic worlds then passed: three steps accepted,
  train +0.094, held-out +0.035 (`outputs/rsi-calibration/cleanroom-v2/`).
- **Smoke is a weak gate.** The untouched policy scores 8/10 on the smoke set (tasks 0 and 8 fail),
  the same as the gate. Only the 100-trial development result separates candidates.
- **First results (smoke only, not conclusions):** harness node, 80.6 % memory removed at 8/10;
  first evidence node `iter_000-b0-a0`, 79.6 % (1.13 GB resident) at 8/10. Method the agent chose:
  drop the unused world model and vocabulary head, shrink the token embedding to the 53 tokens in
  the calibration set, 4-bit quantisation in place. The embedding shrink ties the policy to the
  calibration prompts: say so in the report. Its own check showed about 206 ms median latency; the
  score ignores latency.
- **Evaluator rules added before launch** (protocol `revisions`): workspace files over 1 MiB count
  as memory; memory is the larger of the tensor walk and live CUDA bytes, so a compressed copy
  beside the original scores about 0.
- **Known deviations from the paper** are listed in `docs/DREAM_RSI.md`. Audit:
  `/tmp/opencode/audit/fidelity.md`. Report gap analysis: `/tmp/opencode/audit/report_gap.md`
  (11 of 18 report claims lacked clean-room evidence at 09:30).
- **Dreaming** (replay of candidate policies on recorded trees) first happens when cycle 1 ends.
  With one recorded tree the first selection is overfit by construction.

## Map (read these instead of re-exploring)

`CLAUDE.md` has the directory table and `docs/COMMANDS.md` the one command per task. What they do
not say:

| Need | Go to |
|---|---|
| How the loop differs from the paper | `docs/DREAM_RSI.md`, "Known deviations" |
| What a discovery agent is told | `rsi/prompts/discovery.md`; workspace seed in `rsi/workspace_template/` |
| What the policy-rewrite agent is told | `rsi/prompts/policy_dev.md` |
| How a node is scored | `evaluation/compression/solution.py` (memory rule, violations, time limit) |
| Outer loop, resume, stop rules | `rsi/durable_loop.py`; one cycle in `rsi/online_cycle.py`; one node in `rsi/expansion.py` |
| Replay ("dreaming") and V | `rsi/dream.py`, `rsi/objective.py`, `rsi/policy_dev.py` |
| Launcher, status, stop | `rsi/service.py` |
| Agent CLI wiring (pi) | `rsi/agent_backends.py` |
| Frozen gates, budgets, revisions | `studies/program-v1/protocol.json` |
| A node's record | `studies/program-v1/cycles/<cycle>/nodes/<cell>/{score.json,proposal.md,diff.patch}` |
| Policy versions and replay evidence | `studies/program-v1/policies/` |
| Watch page (old) | `rsi/watch.py`, `rsi/watch_events.py`, `rsi/watch_static/index.html` |
| Watch page (new, unmerged) | UI worktrees; `rsi/watch_static/README.md` and `docs/watch_state.md` there |
| Measurement queue (unmerged) | queue worktree: `evaluation/compression/queue.py`, `studies/program-v1/queue.json`, `recipes/` |
| Scratch from today's agents | `/tmp/opencode/` (audits, UI notes, logs); lost on reboot |

## To check at the end of cycle 1

- Were two agents really running in parallel? After 24 min only one node was done and branch 1 had
  just (re)started.
- Did any node time out or fail for a harness reason rather than a measured one?
- Did policy development produce an accepted step in the real loop?
- Is the measurement queue slowing the agents enough to matter?

## Remaining work, in order

1. Let both runs finish. Review each cycle for prompt or architecture waste.
2. Merge the queue branch; commit Dream-RSI evidence (`cycles/*/tree.json`,
   `live_cycle_manifest.json`, node `score.json`, `proposal.md`, `diff.patch`, `development.json`,
   `logs/command.json`, `policies/*/evidence`, `selection.json`, `selected_policy.py`).
3. Pick winners on development data only, then run Reserved-500 once per winner:
   `python -m evaluation.compression.queue --queue <reserved queue file> --allow-reserved`
   (4 cells, about 260 min as prepared). The C1/C2 joint recipes are not authored yet: they depend
   on development outcomes.
4. `uv run --no-sync vla-report-plots`, check the `report/data/numbers.json` diff, then rewrite the
   report in the order mechanism curves, joint recipes, search, execution.
5. Swap port 8787 to the new watch page once the UI lead's checks pass.
6. Ask Fernando before pushing or posting to the agent board.
