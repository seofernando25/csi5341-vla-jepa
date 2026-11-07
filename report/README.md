# Final report

IEEE two-column, ≤5 pages ([main.pdf](main.pdf)). Every figure and quoted number is generated
from committed evidence under `studies/`.

```bash
uv run --no-sync python report/figures.py   # figures/*.pdf and data/numbers.json
cd report && latexmk -pdf main.tex
```

| File | Role |
|---|---|
| `main.tex`, `refs.bib` | Report source |
| `figures.py` | Builds all figures; prints every reported statistic to `data/numbers.json` |
| `data/parameter_census.json` | Parameter counts by module, read from the checkpoint's safetensors header (cached so the build works without the checkpoint) |

After changing evidence or `figures.py`, compare `data/numbers.json` against the text of
`main.tex` before rebuilding the PDF.
