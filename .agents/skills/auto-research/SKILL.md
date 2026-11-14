---
name: auto-research
description: Run bounded, reproducible Dream-RSI or compression auto-research loops in csi5341-vla-jepa (`/goal`, automated Qwen search rounds, and stop/resume discipline). Use when the user asks to run or automate iterative research rounds over `rsi/`.
---

# Auto Research

Run bounded recursive self-improvement rounds over the open Qwen compression track (`rsi/`, `studies/dream-rsi/`) without touching reserved test trials, overfitting to replay, or burning GPU budgets on unlogged runs.

## Preconditions

1. **Sync & claim:** `git fetch origin main`, read `AGENT_BOARD.jsonl`, verify no active GPU job (`nvidia-smi`), and post a `CLM` (`python scripts/agent_board.py`) with the round ID and hypothesis before editing or launching GPUs.
2. **CPU baseline first:** Run `uv run --no-sync vla-rsi benchmark` and check `studies/dream-rsi/README.md` before designing a new round. Leave-one-world-out replay $V$ chooses the search policy—it does not prove LIBERO task success.
3. **Closed arms stay closed:** Do not restart SmolVLM backbone training/swaps (`82df05d`) without an explicit new experimental plan.

## Bounded Loop

Each round is one self-contained cycle (`<= 6` probes per arm unless budgeted explicitly):

1. **Propose:** Inspect prior `studies/dream-rsi/live_round*.json` failure modes (`smoke`, `dev-100`, task breakdown) and archive updated `OptimalPolicy` revisions (`rsi/develop.py`) or add a bounded round builder (`build_round<N>_candidates` in `rsi/live.py` + `ROUND_BUILDERS` in `rsi/live_round.py`). State the expected size/latency/success trade-off up front.
2. **Budget check:** Declare a hard GPU wall-clock and probe cap before starting (`--budget-each <= 6` for initial screen). One GPU owner at a time.
3. **Run live round:** Execute `uv run --no-sync vla-rsi-round --round <rN> --budget-each <B> --screen-promoted --output studies/dream-rsi/live_round<N>.json` (control vs selected on identical states and seeds).
4. **Gate & record:**
   - Check Wilson 95% CIs, size (MB), and pipeline median/p95 latency cleanly (`reporting-results` checklist).
   - Never pool `screen` (`10`), `development` (`100`), or `final` (`500`) phases.
   - Commit the reproducible JSON journal (`studies/dream-rsi/live_round<N>.json`) and update `studies/dream-rsi/README.md`; never commit raw weights, `.pt` checkpoints, or `outputs/`.

## Hard Locks & Stop Rules

- **Reserved-500 lock:** Never invoke `--phase final` (500 reserved LIBERO trials) inside an automated search loop. Reserved trials are strictly post-search confirmation only after human/explicit promotion of a Pareto candidate.
- **Diminishing-returns stop:** Stop the loop cleanly and publish a summary when either:
  - two consecutive rounds hit a pruning cliff (`smoke < 8/10`) or yield overlapping `dev-100` 95% Wilson CIs with `< 20 MB` / `< 5 ms` Pareto gain, or
  - the declared GPU/probe budget is reached.
- **Resume contract:** `run_live_round` reads `studies/compression/measurements/<cell>/<phase>/run.json` and per-round journals (`outputs/compression/<rN>_recipes/`). On interrupt, inspect `git status` + partial `measurements/` manifests and resume the same `<round_id>` rather than discarding completed cells or mixing seeds across rounds.

## Wrap Up

Before closing the loop:

1. Run `uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"`.
2. If evidence affecting report numbers changed, run `uv run --no-sync vla-report-plots` and confirm `report/data/numbers.json` matches only expected deltas.
3. Push board `DONE` (or `BLK` with exact blocker) referencing the commit SHA and summary metrics.
