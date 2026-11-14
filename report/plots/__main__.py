"""Build all report figures and data/numbers.json from committed study evidence.

Usage: python -m report.plots   (repository root; CPU only, no model loading)
Every number quoted in report/main.tex is written to report/data/numbers.json.
"""

import argparse
import json

from report.plots.data import DATA, collect, divergence, memory_ceiling, parameter_census
from report.plots.divergence import fig_divergence
from report.plots.memory import fig_budget
from report.plots.results import fig_tasks, fig_tradeoff
from report.plots.sensitivity import fig_sensitivity
from report.plots.style import OUT


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    n = collect()
    census = parameter_census()
    n["census"] = census
    n["memory_ceiling"] = memory_ceiling(census, n["rows"]["B16"]["peak_allocated_gb"])
    fig_tradeoff(n)
    fig_budget(census, n["memory_ceiling"], n)
    n["sensitivity"] = fig_sensitivity()
    n["divergence"] = fig_divergence(divergence())
    fig_tasks(n)
    (DATA / "numbers.json").write_text(json.dumps(n, indent=2, default=float) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures and {DATA / 'numbers.json'}")


if __name__ == "__main__":
    main()
