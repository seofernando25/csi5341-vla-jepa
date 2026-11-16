# Handoff

Run state, live branches and unmerged work. Everything else is in `AGENTS.md`, `docs/` and the study
READMEs. Rewrite this file when the state changes; delete it when both runs are merged and reported.
The hour-by-hour log of 2026-11-15 is in Git history (this file at `64206a6`).

## State at 05:15 EST, 2026-11-16

Every run is finished and committed. No GPU job is running.

| Run | Result |
|---|---|
| Reserved-500, v1 selection (6 cells) | reference 95.0 %, cycle-1 winner 93.4 %, cycle-2 winner 95.6 %, lang int8 94.8 %, lang nf4 96.0 %, packed NF4 ladder 97.8 % |
| `program-v2` search (3 cycles, 12 nodes) | did not beat v1: best complete node 79.1 % at 83/100; its 83.4 % node timed out in development |
| Reserved-500, v2 winners (2 cells) | 96.2 % and 95.0 %, both at the reference |

The report is a four-page draft in `report/main.tex`, every number traced to
`report/data/numbers.json`, with four figures and two tables. The v2 worktree and both side branches
are gone; `main` is the only branch.

## State at 08:00 EST, 2026-11-16

- Plan for everything left: `ROADMAP.md` (requirements, figure plan, tasks T1 to T27).
- Running: Dream-RSI `program-v3` (debiased size search: no compression code, no prompt token ids),
  worktree `~/.t3/worktrees/csi5341-vla-jepa/t3-b413d091-e2ebc1e810b892e3`, tag `program-v3-debiased`,
  launched 07:22, ends by about 15:20. Watcher log `/tmp/opencode/s2/J3-STATUS.md`.
- Prepared, not run: latency search `program-v4`, worktree
  `~/.t3/worktrees/csi5341-vla-jepa/t3-ba578c7d-e439f6d8dfe42299`, tag `program-v4-latency`; its timing
  and cache guards are being fixed (`/tmp/opencode/s2/Q2-STATUS.md`). Launch on an idle GPU only.
- Report on `main`: 4 pages, failure analysis and proposal verdict in (`cc1296a`).

## Next, in order

1. Read `report/main.pdf` once for voice and accuracy.
2. Ask Fernando before pushing or posting to the board.
3. Optional: the presentation (`web/presentation/`) still shows the pre-cleanroom story.

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

`main` is the only working branch, and no other worktree exists. `origin/main` is at `ac06d1b`;
nothing since is pushed and no board record is posted. Ask Fernando before either.

## Next, in order

1. When the two v2 reserved cells stop (~05:00): fill the reserved column of Table II in
   `report/main.tex`, rebuild `vla-report-plots`, and commit the records, the figures and
   `report/data/numbers.json`.
2. Update `studies/program-v2/README.md` with the reserved result.
3. Ask Fernando before pushing or posting to the board.

The report draft is committed (`563958b`): four pages, four figures, two tables. Every number in it
comes from `report/data/numbers.json`; the only placeholder is Table II's reserved column.

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
