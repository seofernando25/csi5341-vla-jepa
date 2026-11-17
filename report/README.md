# Final report

`main.tex` is the IEEE two-column report (at most 5 pages). Every number in it comes from
`plots/`, which reads the committed records under `studies/` and writes the figures in `figures/`
and `data/numbers.json`. Run state and open work: [HANDOFF.md](../HANDOFF.md).

```bash
uv run --no-sync vla-report-plots   # rebuild figures and data/numbers.json
cd report && latexmk -pdf main.tex  # build main.pdf (not committed)
```

| Path | Role |
|---|---|
| `main.tex`, `refs.bib` | Report source and bibliography |
| `tex/` | `style.tex` (layout, colours, running header) and `overview.tex` (Fig. 1, TikZ) |
| `plots/` | Figure builders; each builder skips itself when its evidence is absent |
| `figures/` | Generated figure PDFs |
| `data/numbers.json` | Every quoted number, with its source path |
| `data/parameter_census.json` | Parameter counts from the checkpoint header |
| `legacy/` | Old report source; history only, not evidence |
