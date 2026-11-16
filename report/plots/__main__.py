"""Build all report figures and data/numbers.json from committed study evidence.

Usage: python -m report.plots   (repository root; CPU only, no model loading)
Every number quoted in report/main.tex is written to report/data/numbers.json.
A figure whose clean-room evidence is absent is skipped, named on stdout, and recorded under
`figures`; no number from a skipped figure reaches numbers.json.
"""

import argparse
import json

from report.plots.cycles import dream_rsi_cycles
from report.plots.data import (
    DATA,
    STUDY,
    collect,
    memory_ceiling,
    parameter_census,
    plan,
    recovery_dev,
)
from report.plots.divergence import divergence_record, fig_divergence
from report.plots.precision import fig_format, fig_precision
from report.plots.results import fig_ladder
from report.plots.style import OUT


def build(name, builder, *args) -> str:
    """Run one figure builder; report a skip by name instead of failing the whole build."""
    result = builder(*args)
    if result is None:
        print(f"skip {name}: evidence absent")
        return "skipped: evidence absent"
    return "ok"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not (STUDY / "measurements").is_dir():
        print("No clean-room measurements yet; report figures were not rebuilt.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    n = collect()
    census = parameter_census()
    n["census"] = census
    n["plan"] = plan()
    b16 = n["rows"].get("B16") or {}
    if "benchmark_peak_allocated_gb" in b16:
        n["memory_ceiling"] = memory_ceiling(census, b16["benchmark_peak_allocated_gb"])
    n["figures"] = {
        "precision": build("precision", fig_precision, n),
        "format": build("format", fig_format, n),
        "ladder": build("ladder", fig_ladder, n),
    }
    record = divergence_record(STUDY)
    (STUDY / "divergence.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    n["divergence"] = {
        "study": record["study"],
        "counts": record["counts"],
        "phases": record["phases"],
    }
    n["figures"]["divergence"] = build("divergence", fig_divergence, record)
    n["dream_rsi"] = dream_rsi_cycles()
    n["recovery_dev"] = recovery_dev()
    (DATA / "numbers.json").write_text(json.dumps(n, indent=2, default=float) + "\n")
    print(f"Wrote {len(list(OUT.glob('*.pdf')))} figures and {DATA / 'numbers.json'}")


if __name__ == "__main__":
    main()
