"""Build all report figures and data/numbers.json from committed study evidence.

Usage: python -m report.plots   (repository root; CPU only, no model loading)
Every number quoted in report/main.tex is written to report/data/numbers.json.
Figures whose clean-room evidence is absent are skipped and recorded under `figures`.
"""

import argparse
import json

from report.plots.data import (
    DATA,
    STUDY,
    collect,
    divergence,
    dream_rsi_cycles,
    launch_profile,
    memory_ceiling,
    parameter_census,
    recovery_dev,
)
from report.plots.divergence import fig_divergence
from report.plots.results import fig_ladder
from report.plots.sensitivity import fig_sensitivity
from report.plots.style import OUT


def _built(result) -> str:
    return "ok" if result is not None else "skipped: evidence absent"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not (STUDY / "summary.csv").exists():
        print("No clean-room evidence yet; report figures were not rebuilt.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    n = collect()
    census = parameter_census()
    n["census"] = census
    b16 = n["rows"].get("B16") or {}
    if "peak_allocated_gb" in b16:
        n["memory_ceiling"] = memory_ceiling(census, b16["peak_allocated_gb"])
    n["launch_profile"] = launch_profile()
    n["figures"] = {
        "ladder": _built(fig_ladder(n)),
        "sensitivity": _built(fig_sensitivity()),
        "divergence": _built(fig_divergence(divergence())),
    }
    n["dream_rsi"] = dream_rsi_cycles()
    n["recovery_dev"] = recovery_dev()
    (DATA / "numbers.json").write_text(json.dumps(n, indent=2, default=float) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures and {DATA / 'numbers.json'}")


if __name__ == "__main__":
    main()
