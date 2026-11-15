"""Build all report figures and data/numbers.json from committed study evidence.

Usage: python -m report.plots   (repository root; CPU only, no model loading)
Every number quoted in report/main.tex is written to report/data/numbers.json.
"""

import argparse
import json

from report.plots.data import (
    DATA,
    collect,
    divergence,
    dream_rsi_rounds,
    dream_rsi_stage2,
    launch_profile,
    memory_ceiling,
    parameter_census,
    recipe_funnel,
    recovery_dev,
    restricted_vocab_final,
    sensitivity_dev_failures,
)
from report.plots.divergence import fig_divergence
from report.plots.results import fig_ladder
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
    n["launch_profile"] = launch_profile()
    fig_ladder(n)
    dev_fails = sensitivity_dev_failures()
    n["sensitivity"] = fig_sensitivity(dev_fails)
    n["divergence"] = fig_divergence(divergence())
    n["dream_rsi"] = dream_rsi_stage2()
    n["dream_rsi"]["rounds"] = dream_rsi_rounds()
    n["funnel"] = recipe_funnel()
    n["sensitivity_dev_failures"] = dev_fails
    n["recovery_dev"] = recovery_dev()
    n["restricted_vocab_final"] = restricted_vocab_final()
    (DATA / "numbers.json").write_text(json.dumps(n, indent=2, default=float) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures and {DATA / 'numbers.json'}")


if __name__ == "__main__":
    main()
