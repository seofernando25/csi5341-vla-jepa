---
name: auto-research
description: Run bounded, reproducible Dream-RSI or compression auto-research loops in csi5341-vla-jepa (`/goal`, automated Qwen search rounds, and stop/resume discipline). Use when the user asks to run or automate iterative research rounds over `rsi/`.
---

# Auto Research

Run bounded Dream-RSI rounds over the open Qwen compression track (`rsi/`, `studies/program-v1/`) without touching reserved test trials or burning GPU budget on unlogged runs. For `rsi/` internals and paper fidelity, read [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md) and the `dream-rsi` skill.

## Preconditions

1. **Sync and claim:** `git fetch origin main`, read `AGENT_BOARD.jsonl`, confirm with `nvidia-smi` that the GPU is free, and post a `CLM` with the round ID and hypothesis before launching GPUs.
2. **Tests pass:** `uv run --no-sync pytest -q -m "not slow"` is green. This is the only pre-GPU check; there is no fake-environment cycle.
3. **Calibration verdict recorded:** `uv run --no-sync vla-rsi calibrate --tag <name> --prompt rsi/prompts/policy_dev.md` writes `outputs/rsi-calibration/<tag>/` (synthetic worlds, LLM API, no GPU). Record its verdict in the claim. Do not use it as evidence.
4. **Closed arms stay closed:** Do not restart SmolVLM backbone training or swaps (`82df05d`) without a new experimental plan.

## Bounded Loops

Commands are listed in [docs/COMMANDS.md](../../../docs/COMMANDS.md).

The agent backend defaults to the Claude Code CLI, Haiku, low effort (protocol `dream_rsi.agent`). Override with `--backend {claude,pi} --model --effort`. Each manifest records backend, model, effort, CLI version, turns and cost; cite them in the claim.

1. **Single cycle:** `uv run --no-sync vla-rsi cycle --cycle-id iter_000 --probe-budget 8 --promote-top-k 1`
2. **Outer loop** (`plan_grid` -> online `bwrap` agent expansion -> `develop_policies_llm` update): `uv run --no-sync vla-rsi loop --cycles 3 --probe-budget 8 --promote-top-k 1`
3. **Offline policy development only:** `uv run --no-sync vla-rsi develop --cycle-id iter_000 --m 4`
4. **Inspect artifacts:** `studies/program-v1/cycles/<cid>/live_cycle_manifest.json`, `tree.json`, and `policies/<cid>/selection.json`. Declare a GPU wall-clock and probe cap before starting. Measure with Wilson 95% CIs and median/p95 latency. Never pool smoke, development and reserved phases.

## Hard Locks & Stop Rules

- **Reserved-500 lock:** Never invoke `--phase final` (500 reserved LIBERO trials) inside an automated search loop. Reserved trials run only after human or explicit promotion of a Pareto candidate.
- **Diminishing-returns stop:** Stop the loop cleanly and publish a summary when either:
  - two consecutive rounds hit a pruning cliff (`smoke < 8/10`) or yield overlapping development 95% Wilson CIs with `< 20 MB` / `< 5 ms` Pareto gain, or
  - the declared GPU or probe budget is reached.
- **Resume contract:** A node whose `studies/program-v1/cycles/<cid>/nodes/<cell>/score.json` exists is returned as-is and never re-run. `vla-compress run` refuses to overwrite an existing `studies/program-v1/measurements/<recipe>/<phase>/` folder. On interrupt, inspect `git status` and partial manifests, then resume the same `--cycle-id`. Do not discard completed nodes or mix seeds across cycles.

## Wrap Up

1. Run `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"`.
2. If evidence affecting report numbers changed, run `uv run --no-sync vla-report-plots` and confirm `report/data/numbers.json` changed only where expected.
3. Post board `DONE` (or `BLK` with the exact blocker) with the commit SHA and summary metrics.
