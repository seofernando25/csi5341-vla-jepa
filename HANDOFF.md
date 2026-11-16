# Handoff

Live run state only. Everything else is in `CLAUDE.md`, `docs/`, the study READMEs and
`ROADMAP.md`. Rewrite this file when the state changes; delete it when both runs are merged.

## State, 2026-11-16

The report is four pages and committed. `main` is at `62f8e02`, level with `origin/main` before the
current figure and docs work. Next steps and order: `ROADMAP.md` section 4.

| Run | State |
|---|---|
| Reserved-500, v1 selection | complete |
| `program-v2` search | stopped, evidence merged to `main`; do not restart |
| `program-v4` latency search | live since 16:05 UTC. Worktree `/home/fer/.t3/worktrees/csi5341-vla-jepa/t3-run-program-v4`, branch `t3/run-program-v4` (at `0970b90`), unit `rsi-home-fer-t3-worktrees-csi5341-vla-jepa-t3-run-program-v4-studies-program-v4.service`. Node snapshot: `studies/program-v4/nodes.json` |
| `program-v3` debiased size search | paused (`STOP` in its study; not resumed). Worktree `/home/fer/.t3/worktrees/csi5341-vla-jepa/t3-b413d091-e2ebc1e810b892e3`, branch `t3/b413d091`, tag `program-v3-debiased`. Evidence (`cycles/`) is uncommitted there. Not mine; resume only on Fernando's decision |

Neither branch is merged. Merge each when its run ends, then remove the worktree and delete the
branch (ROADMAP T21).

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
  --policy-timeout 1200`. It resumes from `loop_state.json`; the stopped first launch is archived,
  not resumed.
- v3: same command from its worktree with `--study-dir studies/program-v3 --max-parallelism 2`,
  after removing `STOP`.
- Jupyter kernels of `csi5341-a3` stay on the GPU. Never kill another user's processes.
- Keep `~/.pi/agent/models.json` without a `mistral` block. Policy development needs
  `--policy-timeout 1200`.
