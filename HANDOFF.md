# Handoff

Live run state only. Everything else is in `CLAUDE.md`, `docs/`, the study READMEs and
`ROADMAP.md`. Rewrite this file when the state changes; delete it when both runs are merged.

## State, 2026-11-16

The report is four pages and committed on `main`. Next steps and order: `ROADMAP.md` section 4.

| Run | State |
|---|---|
| Reserved-500, v1 selection | complete |
| `program-v2` search | stopped, evidence merged to `main`; do not restart |

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

- Jupyter kernels of `csi5341-a3` stay on the GPU. Never kill another user's processes.
- Keep `~/.pi/agent/models.json` without a `mistral` block. Policy development needs
  `--policy-timeout 1200`.
