# Project scope

CSI 5341 project: post-training compression of the released VLA-JEPA LIBERO policy. Experiments are
complete. Remaining deliverables are the final report (`report/`, IEEE two-column, ≤5 pages, due
2026-12-09) and the final presentation (due 2026-12-04). Keep results attributable to their model,
configuration and protocol.

## Skills

Project skills live in `.agents/skills/` (`.claude/skills` links there):

- `terse-writing`: all prose, including the report, READMEs, commits and board records.
- `reporting-results`: checklist before any number leaves the terminal.
- `agent-board`: reading and posting coordination records.
- `untangle-codebase`: whole-codebase refactor procedure (run only when asked).

## Coordination

`F` = Fernando's agent; `N` = Noah's agent. Follow `AGENT_BOARD.md`; read the latest remote board
before shared work. Append a CLM before shared work and a DONE or BLK after it. Publishing board
records follows the user's communication authorization; never rewrite old records.

## Where things live

- Final report: `report/main.tex`. Figures and every quoted number come from `report/plots/`,
  which reads committed evidence under `studies/`. Rebuild both after any evidence change.
- Study evidence: `studies/<study>/`, indexed in `studies/README.md`. Each study README states
  its question, result, evidence and reproduction command.
- Code: `evaluation/` (rollouts, timing, S500 training, analysis, `fastpath.py` CUDA-graph engine, `nf4.py`), `evaluation/compression/`
  (recipes → transforms → measurements), `src/` (SmolVLM plugin). Closed arms keep only their
  results; their code is in Git history at commit `82df05d`.
- Checks: `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .` and
  `uv run --no-sync pytest` must pass with no warnings.

## Evidence rules

- Phases are separate: smoke (10 trials), development (100) and reserved (500). Reserved trials
  never guide selection. Never pool phases.
- Held-out loss and offline action error are not LIBERO success. Runtime failures are not
  architecture evidence.
- Export compact results with provenance before deleting runtime artifacts.
- Never commit runtime state, weights, datasets, checkpoints, credentials, virtualenvs or
  machine-specific paths. Do not launch training, rollouts or searches during routine code
  validation.
- Qwen is evaluation-only. LeRobot stays pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`.

## Closed arms

The SmolVLM backbone swap (S500), its recovery, the n0008 confirmation and Dream-RSI are closed
negative results. Their summaries stay under `studies/`; do not restart them without a new plan.
