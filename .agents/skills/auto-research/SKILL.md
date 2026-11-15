---
name: auto-research
description: Run bounded, reproducible Dream-RSI or compression auto-research loops in csi5341-vla-jepa (`/goal`, automated Qwen search rounds, and stop/resume discipline). Use when the user asks to run or automate iterative research rounds over `rsi/`.
---

# Auto Research

Bounded Dream-RSI rounds over the clean-room compression track (`rsi/`, `studies/program-v1/`)
without touching reserved trials or burning GPU budget on unlogged runs. Paper fidelity:
[docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md); editing `rsi/` itself: the `dream-rsi` skill.

## Preconditions

1. **Sync and claim:** `git fetch origin main`, read `AGENT_BOARD.jsonl`, confirm with `nvidia-smi`
   that the GPU is free, and post a `CLM` with the round ID and hypothesis before launching GPUs.
2. **Tests pass:** `uv run --no-sync pytest -q -m "not slow"`. This is the pre-GPU check; a cycle can
   also be exercised against a stub with `--evaluator MODULE:FUNC`.
3. **Calibration verdict recorded:** `vla-rsi calibrate --tag <name> --prompt rsi/prompts/policy_dev.md`
   writes `outputs/rsi-calibration/<tag>/` (synthetic worlds, LLM API, no GPU). Record its verdict in
   the claim; it is not evidence.

## Bounded loops

Commands: [docs/COMMANDS.md](../../../docs/COMMANDS.md). The agent backend comes from protocol
`dream_rsi.agent`: pi on `mistral/deepseek-v4-1-flash-1017` at max thinking. Override with
`--backend {claude,pi} --model --effort`. Each manifest records backend, model, effort, CLI version,
turns and cost; cite them in the claim.

1. **Detached run (canonical):** `vla-rsi start --study-dir studies/program-v1 -- --cycles 3
   --probe-budget 8 --promote-top-k 1 --policy-timeout 1200`, then `vla-rsi status`, `vla-rsi watch`,
   `vla-rsi stop`, each with `--study-dir studies/program-v1`.
2. **Single cycle:** `vla-rsi cycle --cycle-id iter_000 --probe-budget 8 --promote-top-k 1`.
3. **Policy development only:** `vla-rsi develop --cycle-id iter_000 --m 4`.
4. **Inspect:** `studies/program-v1/cycles/<cid>/live_cycle_manifest.json`, `tree.json`, and
   `studies/program-v1/policies/<cid>/selection.json`.

Declare a GPU wall-clock and probe cap before starting. Report Wilson 95% CIs and median/p95 latency.
Never pool smoke, development and reserved phases.

## Hard locks and stop rules

- **Reserved-500 lock:** never pass `--phase final` inside a search loop. Reserved trials run only
  after a human freezes one candidate.
- **Diminishing-returns stop:** stop cleanly and publish a summary when two consecutive rounds hit a
  pruning cliff (`smoke < 8/10`) or yield overlapping development 95% Wilson CIs with < 20 MB or
  < 5 ms Pareto gain, or when the declared budget is reached.
- **Resume contract:** a node whose `cycles/<cid>/nodes/<cell>/score.json` exists is returned as-is
  and never re-run. `vla-compress run` refuses to overwrite an existing
  `measurements/<recipe>/<phase>/`. On interrupt, inspect `git status` and partial manifests, then
  resume the same `--cycle-id`.

## Wrap up

Run the checks in AGENTS.md. If evidence affecting report numbers changed, rebuild
`vla-report-plots` and confirm `report/data/numbers.json` changed only where expected. Post board
`DONE` (or `BLK` with the exact blocker) with the commit SHA and summary metrics.
