# Project scope

CSI 5341 project: post-training compression of the released VLA-JEPA LIBERO policy. Deliverables
are the final report (`report/`, IEEE two-column, ≤5 pages, due 2026-12-09) and the final
presentation (due 2026-12-04). Keep results attributable to their model, configuration and protocol.

Current state — what is running, what is blocked, next step: `HANDOFF.md` at the repo root.

- **Open:** Dream-RSI clean-room search over compression solutions (`rsi/`, `studies/program-v1/`).
  A discovery agent writes `proposal.md` and `solution.py` in a per-node, git-snapshotted workspace;
  the fixed evaluator `evaluation/compression/solution.py` scores it; a policy-development agent
  rewrites the exploration policy that replay selects.
- **Measured:** the pre-registered precision surface, run by the measurement queue
  (`evaluation/compression/queue.py`); records under `studies/program-v1/measurements/`.
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
| `evaluation/` | Rollouts, timing, analysis, `fastpath.py` (CUDA-graph engine), `nf4.py`, `fsm.py` (state machines for persisted statuses) |
| `evaluation/compression/` | Recipes and transforms; `solution.py` is the fixed evaluator for an agent-written `solution.py` |
| `rsi/` | Dream-RSI: discovery and policy-development agents, `bwrap` sandbox, replay, gates, live cycles, watch page. Layout and state machines: `rsi/README.md` |
| `web/` | Presentation and explainer sources (`web/README.md`) |
| `scripts/` | Board tooling and small runners |
| `docs/` | `COMMANDS.md`, `DREAM_RSI.md`, `ORCHESTRATION.md`, `COMPRESSION_PROGRAM.md`, README assets |

Every number quoted in the report comes from `report/plots/`, which reads `studies/`. Rebuild both
after any evidence change (`vla-report-plots`) and confirm `report/data/numbers.json` changed only
where intended.

## Code shape

- Reusable code goes in a package; a runner is a thin `main()` exposed as an entry point.
- Aim for files under about 300 lines so one can be read in a sitting. Split at a real seam
  (separate responsibilities), never mechanically. A longer file that does one thing is fine.
- Delete what is superseded. Git history is the archive.

## Branches

- All work lands on `main`, checked out in the main checkout. There is no long-lived side branch.
- A worktree branch exists only for work that cannot touch the main checkout (a live run imports
  it). Merge it when the run or task ends, then remove the worktree and delete the branch the same
  day.
- Run state, live branches and unmerged work are written in `HANDOFF.md` only. The other documents
  describe what is true on `main`.

## Checks before prose

- When an agent has to be corrected, first try to make the mistake impossible or caught: a test, a
  ruff rule, a guard in the evaluator. Add a line here or to a skill only when the point needs
  judgment.
- `tests/test_docs.py` fails when a live document names a missing path, an unknown `vla-*` command
  or transient state such as a branch name. Fix the document; do not weaken the check.
- An agent does not certify its own result. The fixed evaluator, the gates and the reserved
  protocol decide.

## Skills

Project skills live in `.agents/skills/` (`.claude/skills` links there):

- `terse-writing`: all prose, including the report, READMEs, commits and board records.
- `reporting-results`: checklist before any number leaves the terminal.
- `agent-board`: reading and posting coordination records.
- `mastermind`: delegating to subagents and gating their output.
- `orchestrate-and-gate`: multi-step diagnosis and release gates.
- `auto-research`: bounded Dream-RSI compression rounds, stop rules, and reserved-500 lock.
- `dream-rsi`: paper specification (`arXiv:2609.14858v2`), fidelity boundaries (`docs/DREAM_RSI.md`), and `rsi/` invariants.
- `ui-critique`: review of a rendered page, figure or slide.
- `academic-illustration`: house style, TikZ kit and templates for report figures, with the layout mistakes to check.
- `untangle-codebase`: whole-codebase refactor procedure (run only when asked).

## Coordination

`F` = Fernando's agent; `N` = Noah's agent. Follow `AGENT_BOARD.md`; read the latest remote board
before shared work. Append a CLM before shared work and a DONE or BLK after it. Board `--from`
must equal the human owner (F or N, never the other human) and each CLM/DONE/BLK/CORR msg must
start with `[<github-account>/<agent>]` naming the executing agent. Publishing board
records follows the user's communication authorization; never rewrite old records.

Repo orchestration rules (roles, gates, recipes, briefs): `docs/ORCHESTRATION.md`.

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
