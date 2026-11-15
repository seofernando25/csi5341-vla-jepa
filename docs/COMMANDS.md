# Commands

One canonical command per task. Run from the repository root after `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval --extra report`.

| Task | Canonical command | Notes |
|---|---|---|
| Rollout (smoke, 10 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 1 --workers 4` | `development` states, 1/task; GPU |
| Rollout (development, 100 trials) | `uv run --no-sync vla-parallel-rollout run --variant B16 --phase development --episodes 10 --workers 4` | `development` states, 10/task; GPU |
| Rollout (reserved, 500 trials) | `uv run --no-sync vla-eval rollout --variant B16 --phase final --episodes 50` | Official 50/task states; GPU |
| Timing (pipeline + policy) | `uv run --no-sync vla-eval benchmark --variant B16` | Requires `vla-eval collect` bank; GPU |
| Compression recipe | `uv run --no-sync vla-compress run --recipe studies/program-v1/recipes/<id>.json --phase <phase>` | Phases: `screen`, `smoke`, `development`, `final`, `size`, `benchmark` |
| Compression queue | `uv run --no-sync vla-compress-queue --dry-run` then `uv run --no-sync vla-compress-queue` | Ordered cells from `studies/program-v1/queue.json`; resumable; stops on a runtime failure; `--allow-reserved` is required for `queue-reserved.json` |
| Agent-written solution (`compress(policy, calibration)`) | `uv run --no-sync vla-compress solution --workspace DIR --phase {check,smoke,development} --out DIR` | Fixed evaluator; writes `DIR/result.json`, `DIR/log.txt`; GPU |
| Dream-RSI loop | `uv run --no-sync vla-rsi loop --cycles 3 --probe-budget 8 --max-parallelism 2 --promote-top-k 0 --policy-timeout 1200` | Online per-node workspace expansion (discovery agent in `bwrap`, fixed `solution` evaluator in the GPU sandbox) + LLM policy rewriting under `studies/program-v1/`; GPU. Agent flags: `--backend {claude,pi}`, `--model`, `--effort` (default: protocol `dream_rsi.agent`, pi on `mistral/deepseek-v4-1-flash-1017` at max thinking); `--agent-timeout` / `--eval-timeout` default to protocol `dream_rsi.agent_timeout_s` / `eval_timeout_s` (1800 s); `--policy-timeout` defaults to 240 s in the CLI, the protocol records 1200 s, so pass it |
| Dream-RSI detached run | `uv run --no-sync vla-rsi start --study-dir studies/program-v1 -- --cycles 3 --max-hours 8 --max-agent-usd 20 --max-parallelism 2 --promote-top-k 0 --policy-timeout 1200` | Launches `loop` as a `systemd --user` unit (restart on failure; `setsid` fallback); appends `DIR/loop.log`; GPU. Re-running resumes from `DIR/loop_state.json`. `--` separates `start` flags from `loop` flags; write the loop flags only, no `loop` token |
| Dream-RSI status | `uv run --no-sync vla-rsi status --study-dir studies/program-v1 [--json]` | Launcher state, lock holder, stage, nodes done/total, best score, agent cost, log tail; reads only |
| Dream-RSI live watch | `uv run --no-sync vla-rsi watch --study-dir studies/program-v1 [--port 8787] [--host 127.0.0.1]` | Read-only page on loopback: live transcripts, node and policy-dev state; works while the loop runs, finished or not started; no GPU |
| Dream-RSI stop | `uv run --no-sync vla-rsi stop --study-dir studies/program-v1 [--grace 120]` | Touches `DIR/STOP`, waits for clean exit, then stops the unit or process group |
| Calibrate policy-update instructions | `uv run --no-sync vla-rsi calibrate --tag NAME --prompt rsi/prompts/policy_dev.md` | Synthetic worlds only, LLM API, no GPU; writes `outputs/rsi-calibration/<tag>/` (refuses `studies/`) |
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
- `vla-compress` (`evaluation.compression.study:main`): Calibrate (`calibrate`), measure a pre-registered recipe (`run`), and evaluate an agent-written `solution.py` (`solution`).
- `vla-compress-queue` (`evaluation.compression.queue:main`): Run an ordered (recipe, phase) queue from `studies/program-v1/queue.json`; `--dry-run` prints the plan and estimated minutes.
- `vla-compress-analyze` (`evaluation.compression.analysis:main`): Rebuild `studies/program-v1/summary.csv`.
- `vla-compress-cka` (`evaluation.compression.cka:main`): Compute per-block linear CKA profile (`studies/program-v1/cka.json`).
- `vla-rsi` (`rsi.__main__:main`): one online cycle (`cycle`), LLM policy-code rewriting (`develop`), the outer loop (`loop`), its detached launcher (`start`, `status`, `watch`, `stop`), and instruction calibration (`calibrate`).
- `vla-report-plots` (`report.plots.__main__:main`): Rebuild report figures and `report/data/numbers.json`.
