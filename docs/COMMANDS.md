# Commands

One canonical command per task. Run from the repository root after `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval --extra report`.

| Task | Canonical command | Notes |
|---|---|---|
| Rollout (smoke, 10 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 1 --workers 4` | `development` states, 1/task; GPU |
| Rollout (development, 100 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 10 --workers 4` | `development` states, 10/task; GPU |
| Rollout (reserved, 500 trials) | `uv run --no-sync vla-eval rollout --variant B16 --phase final --episodes 50` | Official 50/task states; GPU |
| Timing (pipeline + policy) | `uv run --no-sync vla-eval benchmark --variant B16` | Requires `vla-eval collect` bank; GPU |
| Compression recipe | `uv run --no-sync vla-compress run --recipe studies/compression/recipes/<id>.json --phase <phase>` | Phases: `screen`, `smoke`, `development`, `final`, `size`, `benchmark` |
| Dream-RSI propose | `uv run --no-sync vla-rsi-propose --round r7 --n 6` | Writes `studies/dream-rsi/proposals/<round>.json`; command (`scripts/proposer_opencode.sh`) or OpenAI HTTP; CPU |
| Dream-RSI live round | `uv run --no-sync vla-rsi-round --round r1 --output studies/dream-rsi/live_round1.json` | Control vs selected on `r1`–`r6` or proposed `<round>`; GPU |
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
- `vla-rsi` (`rsi.__main__:main`): CPU replay benchmark (`benchmark`) and Stage 2 Qwen search (`qwen-stage2`).
- `vla-rsi-propose` (`rsi.propose:main`): Propose and strictly validate bounded Dream-RSI candidates into `studies/dream-rsi/proposals/<round>.json` (loads repo-root `.env` if present; real env wins). Exactly one transport per call: command (`--proposer-cmd` / `RSI_PROPOSER_CMD`, e.g. `scripts/proposer_opencode.sh` reading stdin and `$RSI_PROPOSER_MODEL`) or OpenAI-compatible HTTP (`--base-url` / `RSI_PROPOSER_BASE_URL`, e.g. `https://openrouter.ai/api/v1` or `http://127.0.0.1:1234/v1`, plus `--model` / `RSI_PROPOSER_MODEL`, key from `--api-key-env` defaulting to `RSI_PROPOSER_API_KEY`, `--temperature`, `--timeout`; no `max_tokens` override sent). Supports `--dry-run` and `--force`.
- `vla-rsi-round` (`rsi.live_round:main`): Run one live Dream-RSI round (`r1`–`r6` or any round journaled under `studies/dream-rsi/proposals/<round>.json`).
- `vla-report-plots` (`report.plots.__main__:main`): Rebuild report figures and `report/data/numbers.json`.
