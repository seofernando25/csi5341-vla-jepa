# Project scope

CSI 5341 project: post-training compression of the released VLA-JEPA LIBERO policy. Deliverables
are the final report (`report/`, IEEE two-column, ≤5 pages, due 2026-12-09) and the final
presentation (due 2026-12-04). Keep results attributable to their model, configuration and protocol.

- **Open:** Dream-RSI search over Qwen compression recipes (`rsi/`, `studies/dream-rsi/`).
- **Closed, negative:** the SmolVLM backbone swap (S500), its recovery and the n0008 confirmation.
  Their summaries stay under `studies/`; their code is in Git history at `82df05d`. Do not restart
  them without a new plan.

## Commands

`docs/COMMANDS.md` is the single command table: one canonical command per task, each a `vla-*`
entry point from `pyproject.toml`. Add a row there when you add a runner; do not add a second way
to do the same thing.

Checks, required before every commit and run by CI:

```sh
uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q
```

They must pass with no warnings. Mark slow or GPU tests `@pytest.mark.slow`.

## Where things live

| Path | Holds |
|---|---|
| `report/` | `main.tex`; `plots/` builds every figure and `data/numbers.json` from committed evidence |
| `studies/<study>/` | Evidence. Each has a README (question, result, evidence, reproduction command), compact tables the report reads, and raw per-run records. Indexed in `studies/README.md` |
| `evaluation/` | Rollouts, timing, analysis, `fastpath.py` (CUDA-graph engine), `nf4.py` |
| `evaluation/compression/` | Recipes → transforms → measurements |
| `rsi/` | Dream-RSI: replay, gates, live rounds |
| `web/` | Presentation and explainer sources (`web/README.md`) |
| `scripts/` | Board tooling and small runners |
| `docs/` | `COMMANDS.md`, README assets |

Every number quoted in the report comes from `report/plots/`, which reads `studies/`. Rebuild both
after any evidence change (`vla-report-plots`) and confirm `report/data/numbers.json` changed only
where intended.

## Code shape

- Reusable code goes in a package; a runner is a thin `main()` exposed as an entry point.
- Aim for files under about 300 lines so one can be read in a sitting. Split at a real seam
  (separate responsibilities), never mechanically. A longer file that does one thing is fine.
- Delete what is superseded. Git history is the archive.

## Skills

Project skills live in `.agents/skills/` (`.claude/skills` links there):

- `terse-writing`: all prose, including the report, READMEs, commits and board records.
- `reporting-results`: checklist before any number leaves the terminal.
- `agent-board`: reading and posting coordination records.
- `mastermind`: delegating to subagents and gating their output.
- `orchestrate-and-gate`: multi-step diagnosis and release gates.
- `auto-research`: bounded Dream-RSI compression rounds, stop rules, and reserved-500 lock.
- `untangle-codebase`: whole-codebase refactor procedure (run only when asked).

## Coordination

`F` = Fernando's agent; `N` = Noah's agent. Follow `AGENT_BOARD.md`; read the latest remote board
before shared work. Append a CLM before shared work and a DONE or BLK after it. Board `--from`
must equal the human owner (F or N, never the other human) and each CLM/DONE/BLK/CORR msg must
start with `[<github-account>/<agent>]` naming the executing agent. Publishing board
records follows the user's communication authorization; never rewrite old records.

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
