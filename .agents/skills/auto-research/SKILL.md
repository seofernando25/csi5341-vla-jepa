---
name: auto-research
description: Run bounded, reproducible Dream-RSI or compression auto-research loops in csi5341-vla-jepa (`/goal`, automated Qwen search rounds, and stop/resume discipline). Use when the user asks to run or automate iterative research rounds over `rsi/`.
---

# Auto Research

Run bounded recursive self-improvement rounds over the open Qwen compression track (`rsi/`, `studies/dream-rsi/`) without touching reserved test trials, overfitting to replay, or burning GPU budgets on unlogged runs. For `rsi/` internals, paper fidelity boundaries (`arXiv:2609.14858v2`), and prefix-only invariants, read [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md) and the `dream-rsi` skill.

## Preconditions

1. **Sync & claim:** `git fetch origin main`, read `AGENT_BOARD.jsonl`, verify no active GPU job (`nvidia-smi`), and post a `CLM` (`python scripts/agent_board.py`) with the round ID and hypothesis before editing or launching GPUs.
2. **CPU baseline first:** Run `uv run --no-sync vla-rsi benchmark` and check `studies/dream-rsi/README.md` before designing a new round. Leave-one-world-out replay $V$ chooses the search policy—it does not prove LIBERO task success.
3. **Closed arms stay closed:** Do not restart SmolVLM backbone training/swaps (`82df05d`) without an explicit new experimental plan.

## Bounded Loops

### Dream-RSI (`studies/dream-rsi-v2/`)

1. **Dry / CPU check:** Confirm `bwrap` and `pi` work cleanly (`uv run --no-sync pytest -q tests/test_dream_rsi_v2.py`) and ensure `AGENT_BOARD.jsonl` shows no competing GPU owner.
2. **Run bounded cycle or outer loop:**
   - Single cycle: `uv run --no-sync vla-rsi cycle --cycle-id iter_000 --probe-budget 8 --promote-top-k 1`
   - Full outer loop (`plan_grid` -> online `bwrap`+`pi` expansion -> `develop_policies_llm` monotonic update): `uv run --no-sync vla-rsi loop --cycles 3 --probe-budget 8 --promote-top-k 1`
   - Offline policy dev only: `uv run --no-sync vla-rsi develop --cycle-id iter_000 --m 4`
3. **Inspect artifacts:** `studies/dream-rsi-v2/cycles/<cid>/live_cycle_manifest.json`, `tree.json`, and `policies/<cid>/selection.json`. Resume uses cached per-node `eval.json` automatically; to restart fresh, use a clean `--study-dir` (or clear the `iter_*` directories and matching `measurements/iter_*`).

### Past Rounds (`studies/dream-rsi/`)

Committed journals under `studies/dream-rsi/` are replay-only evidence for `vla-rsi benchmark`; the round-runner code that produced them was removed (reproducible from git history). New rounds run under `studies/dream-rsi-v2/` as above: declare a GPU wall-clock and probe cap, run `cycle` or `loop`, then gate & record (Wilson 95% CIs, MB, median/p95 latency; never pool `screen`/`development`/`final`).

## Hard Locks & Stop Rules

- **Reserved-500 lock:** Never invoke `--phase final` (500 reserved LIBERO trials) inside an automated search loop. Reserved trials are strictly post-search confirmation only after human/explicit promotion of a Pareto candidate.
- **Diminishing-returns stop:** Stop the loop cleanly and publish a summary when either:
  - two consecutive rounds hit a pruning cliff (`smoke < 8/10`) or yield overlapping `dev-100` 95% Wilson CIs with `< 20 MB` / `< 5 ms` Pareto gain, or
  - the declared GPU/probe budget is reached.
- **Resume contract:** Per-node `eval.json` under `studies/dream-rsi-v2/cycles/<cid>/nodes/` is reused on re-run, and committed measurements under `studies/compression/measurements/<cell>/<phase>/run.json` are never recomputed. On interrupt, inspect `git status` + partial manifests and resume the same `--cycle-id` rather than discarding completed nodes or mixing seeds across cycles.

## Wrap Up

Before closing the loop:

1. Run `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"`.
2. If evidence affecting report numbers changed, run `uv run --no-sync vla-report-plots` and confirm `report/data/numbers.json` matches only expected deltas.
3. Push board `DONE` (or `BLK` with exact blocker) referencing the commit SHA and summary metrics.
