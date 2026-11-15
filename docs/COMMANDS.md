# Commands

One canonical command per task. Run from the repository root after `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval --extra report`.

| Task | Canonical command | Notes |
|---|---|---|
| Rollout (smoke, 10 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 1 --workers 4` | `development` states, 1/task; GPU |
| Rollout (development, 100 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 10 --workers 4` | `development` states, 10/task; GPU |
| Rollout (reserved, 500 trials) | `uv run --no-sync vla-eval rollout --variant B16 --phase final --episodes 50` | Official 50/task states; GPU |
| Timing (pipeline + policy) | `uv run --no-sync vla-eval benchmark --variant B16` | Requires `vla-eval collect` bank; GPU |
| Compression recipe | `uv run --no-sync vla-compress run --recipe studies/compression/recipes/<id>.json --phase <phase>` | Phases: `screen`, `smoke`, `development`, `final`, `size`, `benchmark` |
| Dream-RSI loop | `uv run --no-sync vla-rsi loop --cycles 3 --probe-budget 8` | Online `bwrap`+`pi` per-node expansion + LLM policy rewriting under `studies/dream-rsi-v2/`; GPU |
| Rebuild report plots | `uv run --no-sync vla-report-plots` | CPU only; writes `report/figures/*.pdf` and `report/data/numbers.json` |
| Build report PDF | `cd report && latexmk -pdf main.tex` | Writes `report/main.pdf` |
| Build presentation | `python3 web/presentation/tools/assemble.py` | After browser authoring export; serve with `python3 -m http.server 8000 --bind 127.0.0.1` |
| Checks | `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q` | Fast CPU checks: `pytest -q -m "not slow"` |

## Additional CLI entry points

All scripts accept `--help` without running GPU jobs or modifying outputs:

- `vla-eval` (`evaluation.run:main`): Study A observation collection (`collect`), serial rollouts (`rollout`), and pipeline timing (`benchmark`).
- `vla-parallel-rollout` (`evaluation.parallel_rollout:main`): Multi-worker LIBERO rollouts (`run`) and stage profiler (`profile`).
- `vla-core-benchmark` (`evaluation.core_benchmark:main`): Neural-only timing after `1e-5` equivalence check.
- `vla-launch-profile` (`evaluation.launch_profile:main`): CUDA kernel launch and host-sync profile.
- `vla-eval-analyze` (`evaluation.analysis:main`): Summarize Study A runs into `summary.json` and `T1_summary.csv`.
- `vla-hardware` (`evaluation.hardware:main`): Capture portable hardware inventory JSON.
- `vla-sources` (`evaluation.sources:main`): Download pinned model weights and simulator assets.
- `vla-compress` (`evaluation.compression.study:main`): Calibrate (`calibrate`) and measure (`run`) Study B compression recipes.
- `vla-compress-analyze` (`evaluation.compression.analysis:main`): Rebuild `studies/compression/summary.csv`.
- `vla-compress-cka` (`evaluation.compression.cka:main`): Compute per-block linear CKA profile (`studies/compression/cka.json`).
- `vla-rsi` (`rsi.__main__:main`): CPU replay benchmark (`benchmark`), online cycle (`cycle`), LLM policy-code rewriting (`develop`), and full outer loop (`loop`).
- `vla-report-plots` (`report.plots.__main__:main`): Rebuild report figures and `report/data/numbers.json`.
