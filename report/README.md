# Final report

Clean-room reset 2026-11-15. The old report source and generated artifacts were moved out of the
active path. The old source is at [legacy/main.tex](legacy/main.tex) for history only; do not quote
its numbers.

The active [main.tex](main.tex) is a scaffold. It has no results until the clean runs land under
`studies/program-v1/`.

```bash
uv run --no-sync python -m report.plots   # only after program-v1 evidence exists
cd report && latexmk -pdf main.tex
```

| File | Role |
|---|---|
| `main.tex`, `refs.bib` | Active report skeleton |
| `legacy/` | Old report source; history only, not evidence |
| `plots/` | Figure builders; must be rewritten for program-v1 |
| `data/parameter_census.json` | Parameter counts from the checkpoint header; not a result |

Every number in the active report must trace to a committed record under `studies/program-v1/`.
