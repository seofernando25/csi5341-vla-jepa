# Orchestration

How this repo runs multi-agent work: who decides, who checks, what gets spawned. Project
invariants live in [AGENTS.md](../AGENTS.md); delegation mechanics live in the `mastermind`
skill. This file holds the repo's orchestration rules.

## 1. When to orchestrate

Match effort to risk. Trivial, reversible edits: solo. Grunt work — greps, inventories,
rebuilds, fact-checks — goes to cheap children. Decisions, gates and shared state stay with
the orchestrator. A task that fits in one sitting does not need a swarm.

## 2. Roles

- **Proposer** — writes independent options; never sees the other proposer's draft.
- **Judge** — reviews under one lens: hostile reviewer, outsider reader, or editor for voice.
- **Fuser** — merges judged proposals into one decision and records why alternatives lost.
- **Scout** — sweeps one subsystem, reports findings, fixes nothing.
- **Checker** — reproduces a finding, classifies REAL / LATENT / FALSE, proposes fix and test.
- **Confirmer** — re-runs the checker's evidence independently before it counts.

## 3. Gates

Write kill criteria before results exist; a gate that moves after the fact is not a gate.
Every claim is independently re-verified. Every number traces to an evidence key.

- **Advocate gate.** Before spending effort on a new idea, one child argues for it and one
  against, from the same evidence. Kill criteria are written before the debate starts.
- **Proposal gate.** 2 independent proposers → 3 judges (hostile, outsider, editor) → 1 fuser.
  Judges see the proposals, not each other's verdicts. The fuser owns the final call.

## 4. Recipes

- **Proposal:** 2P → 3J → 1F, for any decision with real alternatives.
- **Bug hunt:** one scout per subsystem → one checker per finding → confirmer re-runs.
- **Report pass:** fact agents trace each number to an evidence key (`[must-fix]` / `[nice]`);
  voice agents cut AI tells and restated numbers and explain metrics in plain words; visual
  agents inspect the render.
- **Visual pass:** render at print size; check greyscale, colorblind safety and ≥7 pt figure
  text. Direct labels over legends. Let figures carry numbers, not prose.
- **Cleanup:** inventory → export compact results with provenance → delete → rebuild → diff.
  The export exists before anything is deleted.

## 5. Brief template

Every child gets a written brief:

- Objective and hard constraints, including what it may not touch.
- Input files with paths and line ranges.
- Deliverable: exact path, format, word or line cap.
- Verification commands it must run itself.
- Report length cap.

## 6. Routing and budget

Tiers and models: the `mastermind` skill. Children do the grep, inventory, rebuild and
fact-check work; the orchestrator never does grunt work. Cap rounds up front. A failed gate
ends the round; it does not buy a retry.

## 7. Failure modes

One guard each:

- **Self-review** — the author never grades its own work; gate per the `mastermind` skill.
- **Number drift** — rebuild `report/data/numbers.json` and diff before quoting a number.
- **Provenance-less deletion** — export compact results with provenance first (recipe above).
- **Reserved-trial leakage** — selection briefs list only smoke and development inputs; the
  reserved lock is defined in AGENTS.md.
- **Page overflow** — check the page budget after every report edit, not at the end.

## 8. Standards and pointers

- Invariants — phases, reserved lock, pins, checks, attribution: [AGENTS.md](../AGENTS.md).
  This doc adds no exceptions.
- Commands: `docs/COMMANDS.md`, one canonical command per task.
- Report numbers: `report/plots` → `report/data/numbers.json` (rule: AGENTS.md).
- Delegation mechanics: `.agents/skills/mastermind/SKILL.md`.
- Owners: Fernando Nogueira (`seofernando25`, compression study and report) and Noah Sprenger
  (`n123xyz`, Dream-RSI). Coordinate through the board per AGENTS.md; credit both in shared
  records.