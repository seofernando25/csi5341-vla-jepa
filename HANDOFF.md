# Handoff

Live run state only. Everything else is in `CLAUDE.md`, `docs/`, the study READMEs and
`ROADMAP.md`. Rewrite this file when the state changes; delete it when both runs are merged.

## State, 2026-11-17

No run is live. The report is complete at five pages on `main`: three searches for memory, one for
speed, each winner run on the reserved 500. Open: Noah's acknowledgment (T19), the final read by
both authors (T20), the presentation (T26, due 2026-12-04).

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
record; a half-finished cell is resumed or reported, never restarted on different states. One exception, disclosed in
`studies/program-v3/README.md`: our tooling killed a v3 reserved cell after 123 episodes, all
successes; it was rerun once from the start and the partial record is kept.

## If the machine rebooted

- Jupyter kernels of `csi5341-a3` stay on the GPU. Never kill another user's processes.
- Keep `~/.pi/agent/models.json` without a `mistral` block. Policy development needs
  `--policy-timeout 1200`.
