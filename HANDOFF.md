# Handoff

Run state, live branches and unmerged work. Everything else is in `AGENTS.md`, `docs/` and the study
READMEs. Rewrite this file when the state changes; delete it when both runs are merged and reported.
The hour-by-hour log of 2026-11-15 is in Git history (this file at `64206a6`).

## State at 03:00 EST, 2026-11-16

`program-v2` stopped cleanly at 02:45 after three cycles: 12 nodes, 5.4 h, $2.32. It did not beat
program-v1. Its best node with a complete development record removed 79.1 % at 83/100, against
v1's 83.0 % at 80/100. Its highest-compression node (83.4 %) timed out in development at 59/100
and is not selectable. The result is written in `studies/program-v2/README.md`.

Reserved-500 for the v1 selection is on its last cell (packed NF4 ladder), expected ~03:35. Five
cells are complete: reference 95.0 %, cycle-1 winner 95.6 %, cycle-2 winner 93.4 %, int8 94.8 %,
nf4 96.0 %.

| Run | Where | State |
|---|---|---|
| Reserved-500 v1, 6 cells | main checkout; log `/tmp/opencode/s2/reserved.log`; records in `studies/program-v1/reserved/` and `studies/program-v1/measurements/*/final/` (uncommitted) | last cell running, ends ~03:35 |
| `program-v2` search | worktree `~/.t3/worktrees/csi5341-vla-jepa/t3-32b33602-0c4d638784cb7753`; evidence committed on its branch | stopped, merge pending |
| Reserved-500 v2, 2 cells | not started; workspaces exported to `/tmp/opencode/s2/v2-ws/` | after the merge, ~80 min each |

Do not edit `evaluation/` in the main checkout until the last v1 cell stops; it imports that code.
Documents and tests are safe. Watch page for v2: port 8899.

## Project explainer video (done, committed to main)

`web/project-explainer/` is the tracked deliverable: 240 s, 1280×720 @ 30 fps,
`dist/project-explainer-720p30.mp4`, h264 High/yuv420p + AAC stereo, ~35 MB.
Direct Skia render pipeline (`src/render.mjs`, frames stream raw into ffmpeg; ~50 s/render).
Four cinematic 3D scenes (Three.js world in `src/world3d/`, alpha PNG windows at 30 fps,
deterministic pick-and-place of the bowl onto the plate, composited with 2D overlays):
`arm` 10.6–21.7 s, `policy` 22.3–33.7 s, `success` 34.3–43.4 s, `episode` 44.0–61.4 s.
Flat scenes are borderless (no cards). `build/3d/` is gitignored; re-render with
`bun tools/render3d.mjs && bun src/render.mjs`.

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

1. When the ladder cell stops (~03:35): merge `t3/implement-brief-k` (clean) and
   `t3/brief-e-implementation` (two conflicts, in `evaluation/compression/solution.py` and
   `study.py`; keep both sides: main's reserved-phase gate and the branch's `--protocol` override).
   Run the checks. Remove both worktrees and their branches.
2. Write `studies/program-v2/freeze.json` (draft in `/tmp/opencode/s2/freeze-v2.json`) and commit
   it before any v2 reserved episode.
3. Run `/tmp/opencode/s2/run_reserved_v2.sh`: `iter_002-b2-a2`, then `iter_001-b2-a1`, from the
   exported workspaces. The reference's reserved result is reused: same states and seeds.
4. Gate the reserved table (development and reserved in separate columns), then commit the records
   and the study README table.
5. `vla-report-plots`, check the `report/data/numbers.json` diff, write the report: mechanism
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

- v2: it stopped; the evidence is committed on `t3/brief-e-implementation` and merged to `main`.
  Do not restart it. Its reserved cells are `/tmp/opencode/s2/run_reserved_v2.sh`.
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
