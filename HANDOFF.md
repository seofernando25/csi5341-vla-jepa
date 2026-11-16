# Handoff

Run state, live branches and unmerged work. Everything else is in `AGENTS.md`, `docs/` and the study
READMEs. Rewrite this file when the state changes; delete it when both runs are merged and reported.
The hour-by-hour log of 2026-11-15 is in Git history (this file at `64206a6`).

## State at 11:15 EST, 2026-11-16

The report is four pages and committed; the independent review's blockers and major findings are
fixed. `main` is pushed to `origin/main` at `8792c14`.

| Run | State |
|---|---|
| Reserved-500, v1 selection (6 cells) | complete: reference 95.0 %, cycle-1 winner 95.6 %, cycle-2 winner 93.4 %, lang int8 94.8 %, lang nf4 96.0 %, packed NF4 ladder 97.8 % |
| `program-v2` search | stopped; did not beat v1; reserved 96.2 % and 95.0 % |
| `program-v4` latency search | first cycle stopped because the score was gamed; fixed; rerun live since 16:05 UTC; first node `iter_000-b0-a0` scores 2.706 (point 2.741, all ten tasks 2.70-2.79, smoke 8/10, no violations), a general CUDA-graph executor; second node running |
| `program-v3` debiased size search | paused; worktree `t3/b413d091`, tag `program-v3-debiased`, `STOP` in its study; not resumed |

The v4 fix is in `d0ca908`: every task is timed (18 calls per task, six blocks of three), the
speedup is the geometric mean over all ten tasks, and the equivalence, arm-action and perturbation
guards are judged per task with the worst task deciding. The stopped cycle is archived in
`studies/program-v4/stopped/`; its task-0 node scored 3.40 under the old three-task score. Identity
under the revised score: 0.987 (point 0.998).

## Next, in order

1. When v4 stops: select on development only, run the reserved 500, add the result to the report.
2. The independent review is done; its blockers and major findings are fixed in `8792c14`. Read
   `report/main.pdf` once more for voice.
3. Push the v4 branch and merge it to `main` when the run and the report are done.
4. Optional: the presentation (`web/presentation/`) still shows the pre-cleanroom story.

## Branches

`main` is at `8792c14` and pushed. Two worktrees exist: the paused `program-v3` run
(`t3/b413d091`, tag `program-v3-debiased`) and the live `program-v4` run
(`t3/run-program-v4`, at `0970b90`). Neither is merged. The v3 run is not mine; do not restart it
without a decision from Fernando.

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

- v4: from its worktree, `uv run --no-sync vla-rsi start --study-dir studies/program-v4 -- --cycles 3
  --probe-budget 8 --max-parallelism 1 --promote-top-k 1 --max-hours 8 --max-agent-usd 20
  --policy-timeout 1200`. It resumes from `loop_state.json`; the stopped cycle is archived, not
  resumed.
- v2: it stopped; the evidence is committed and merged to `main`. Do not restart it. Its reserved
  cells are `/tmp/opencode/s2/run_reserved_v2.sh`.
- Reserved: `outputs/session-s2/run_reserved.sh` and `outputs/session-s2/G.md`.
- Briefs and status files of the 2026-11-15 agents: `outputs/session-s2/` (the live copies in
  `/tmp/opencode/s2/` are lost on reboot).

## Known weak points

- The open policy agent passed one synthetic step before launch, not a live cycle. The
  anytime-salvage path has CPU tests only. v2's identity check was skipped (its baseline node is
  the check).
- v1 cycle-3 nodes `iter_002-b1-a3`, `iter_002-b2-a1`, `iter_002-b2-a2` are the exploit: invalid,
  never reported or plotted as results.
- v4 cycle 1 is the second exploit: a task-0-only speedup under a three-task score. It is archived
  in `studies/program-v4/stopped/` and never selected.
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

## Project explainer video (done, committed to main)

`web/project-explainer/` is the tracked deliverable: 240 s, 1280×720 @ 30 fps,
`dist/project-explainer-720p30.mp4`, h264 High/yuv420p + AAC stereo, ~35 MB.
Direct Skia render pipeline (`src/render.mjs`, frames stream raw into ffmpeg; ~50 s/render).
Four cinematic 3D scenes (Three.js world in `src/world3d/`, alpha PNG windows at 30 fps,
deterministic pick-and-place of the bowl onto the plate, composited with 2D overlays):
`arm` 10.6–21.7 s, `policy` 22.3–33.7 s, `success` 34.3–43.4 s, `episode` 44.0–61.4 s.
Flat scenes are borderless (no cards). `build/3d/` is gitignored; re-render with
`bun tools/render3d.mjs && bun src/render.mjs`.
