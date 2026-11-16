# Handoff

Run state, live branches and unmerged work. Everything else is in `AGENTS.md`, `docs/` and the study
READMEs. Rewrite this file when the state changes; delete it when both runs are merged and reported.
The hour-by-hour log of 2026-11-15 is in Git history (this file at `64206a6`).

## State at 01:20 EST, 2026-11-16

Two runs are live on the one RTX 3090. Do not edit code in the main checkout (the reserved run
imports it) or touch the v2 worktree until they stop. Documents and tests are safe to edit.

| Run | Where | Unit | Expected end |
|---|---|---|---|
| Reserved-500, 6 frozen cells: B16, `iter_001-b1-a3`, `iter_000-b0-a3`, lang INT8, lang NF4, NF4 graph 1-step packed | main checkout; log `/tmp/opencode/s2/reserved.log`; records in `studies/program-v1/reserved/` and `studies/program-v1/measurements/*/final/` (uncommitted) | `vla-reserved` | 04:30 to 05:30 |
| Dream-RSI `program-v2` (open policy agent, anytime steps, live `plan_grid`, fixed memory rule, one GPU slot) | worktree `~/.t3/worktrees/csi5341-vla-jepa/t3-32b33602-0c4d638784cb7753`, tag `program-v2-open`; evidence in its `studies/program-v2/` | `rsi-…-studies-program-v2` | 04:00 to 05:00 (8 h cap) |

The first three reserved cells are done; the int8 anchor is running. v2 finished cycles 0 and 1 and
is in policy development before its last cycle. Watch page for v2: port 8899.

## Project explainer video (done, merged to main)

`web/project-explainer/` is the tracked deliverable: 240 s, 1280×720 @ 30 fps,
`dist/project-explainer-720p30.mp4`, h264 High/yuv420p + AAC stereo, ~35 MB.
Direct Skia render pipeline (`src/render.mjs` — no browser, no screenshots, frames stream raw
into ffmpeg; ~5 ms/frame). Scenes fade in over 0.24 s so the whip never shows a washed-out
scene. `web/dream-rsi-explainer/` is superseded and deleted; the watch page is unchanged and
points at `studies/program-v2`.

## Branches

`main` is the only working branch. `origin/main` is at `ac06d1b`; nothing since is pushed and no
board record is posted. Ask Fernando before either.

| Branch | Worktree | Holds | Action |
|---|---|---|---|
| `t3/brief-e-implementation` | `t3-32b33602-…` | open policy agent, `program-v2` code and evidence | merge after v2 stops, absolute paths stripped; then remove worktree and branch |
| `t3/implement-brief-k` | `t3-5e9d9170-…` | divergence analysis (2 commits, untracked `report/data/numbers.json` and figures) | gate, merge or drop; then remove |

Seven merged side branches and their worktrees were removed on 2026-11-16. Their ignored runtime
files are archived in `outputs/worktree-archive/`.

## Next, in order

1. Reserved queue finishes: int8, nf4, packed ladder. Gate the table (development and reserved in
   separate columns), then commit records and the study README table.
2. v2 runs its last cycle and stops. Audit each promoted node for memory tricks, pick winners on
   development data only, write a v2 freeze record, then run Reserved-500 on them (about 1.5 h
   each). The reference's reserved result is reused: same states.
3. Merge the two branches above.
4. `vla-report-plots`, check the `report/data/numbers.json` diff, write the report: mechanism
   curves, Dream-RSI discoveries, the gaming subsection (already in `report/main.tex`), execution.

## Decisions Fernando made (2026-11-15)

- The report is centred on Dream-RSI discoveries. Every result is measured honestly; bad or missing
  results are reported as such.
- The policy-development agent is open by default (shell, GPU, installs, own replay). v2 is the
  main search result. v1 stays as the record of the memory-score exploit
  (`studies/program-v1/exploit/`) and as a floor: its two winners are real compressions.
- Hand-authored joint recipes C1/C2 are dropped; Dream-RSI winners take that place.
- The scalar score stays: no vector score, no LLM judge.
- A divergence analysis is an optional extra.
- Jupyter kernels of `csi5341-a3` stay on the GPU. Never kill another user's processes.

## Promises to keep

Reserved never guides selection. If v2 ends at two cycles or does worse than v1, the report says so.
No substitute measurement is passed off as the planned one. Never rerun a reserved cell that has a
record; a half-finished cell is resumed or reported, never restarted on different states.

## If the machine rebooted

- v2: from its worktree, `uv run --no-sync vla-rsi start --study-dir studies/program-v2 -- --cycles 3
  --probe-budget 8 --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20
  --policy-timeout 1200`. It resumes from `loop_state.json`.
- Reserved: `outputs/session-s2/run_reserved.sh` and `outputs/session-s2/G.md`.
- Briefs and status files of the 2026-11-15 agents: `outputs/session-s2/` (the live copies in
  `/tmp/opencode/s2/` are lost on reboot).

## Known weak points

- The open policy agent passed one synthetic step before launch, not a live cycle. The
  anytime-salvage path has CPU tests only. v2's identity check was skipped (its baseline node is
  the check).
- v1 cycle-3 nodes `iter_002-b1-a3`, `iter_002-b2-a1`, `iter_002-b2-a2` are the exploit: invalid,
  never reported or plotted as results.
- v1 policy selection is in-sample only and overfit: held-out V fell in cycle 2.
- Smoke is a weak gate: the untouched policy scores 8/10, the same as the gate.
- The embedding shrink some nodes use ties the policy to the calibration prompts. Say so in the
  report.
- `vla-rsi status` prints `agent_cost_usd: 0.0`; the node records hold the real cost.
- GPU is shared, so all times are estimates.

## Facts worth knowing

- Agent: pi on `mistral/deepseek-v4-1-flash-1017`, max thinking. A discovery node takes 14 to 20 min
  and about $0.10. Keep `~/.pi/agent/models.json` without a `mistral` block.
- Policy development needs `--policy-timeout 1200`; at the 240 s CLI default every step timed out.
- Stale user unit `csi5341-rsi-disk-guard.service` points at a path that does not exist; it is
  disabled and can be ignored.

## Map

| Need | Go to |
|---|---|
| How the loop differs from the paper | `docs/DREAM_RSI.md`, "Known deviations" |
| What the agents are told | `rsi/prompts/discovery.md`, `rsi/prompts/policy_dev.md`; workspace seed in `rsi/workspace_template/` |
| How a node is scored | `evaluation/compression/solution.py` |
| Outer loop, one cycle, one node | `rsi/durable_loop.py`, `rsi/online_cycle.py`, `rsi/expansion.py` |
| Replay ("dreaming") and V | `rsi/dream.py`, `rsi/objective.py`, `rsi/policy_dev.py` |
| Launcher, status, stop | `rsi/service.py` |
| Frozen gates, budgets, revisions | `studies/program-v1/protocol.json`, `studies/program-v1/freeze.json` |
| A node's record | `studies/program-v1/cycles/<cycle>/nodes/<cell>/` |
| Measurement queue | `evaluation/compression/queue.py`, `studies/program-v1/queue.json` |
| Watch page | `rsi/watch.py`, `rsi/watch_static/README.md`, `docs/watch_state.md` |
