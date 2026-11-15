---
name: mastermind
description: Fan a job out to cheap subagents and gate what comes back. Use when the user says /mastermind, asks to delegate, parallelise or "use subagents", or when a task splits into independent pieces.
---

# Mastermind

The orchestrator plans, briefs, verifies and decides. Subagents do the reading, computing and
editing.

Repo orchestration rules — roles, gates, recipes, brief contents, standards — live in
`docs/ORCHESTRATION.md`; project invariants in `AGENTS.md`. This skill covers delegation
mechanics only.

## Loop

1. **Split.** One deliverable per child, checkable without reading the child's transcript: a
   file, a JSON answer, a passing command. Pieces that touch the same file or the same
   exclusive resource (a device, a GPU, a deploy target) are one piece with one owner.
2. **Route** each piece by tier (below). Start at the cheapest tier that fits; escalate only
   after a failed check.
3. **Brief.** Write the brief to a file and tell the child to read it; contents follow
   `docs/ORCHESTRATION.md` §5. Give paths and line ranges, not "find the code".
4. **Launch async**, all independent children in one message. Do not poll.
5. **Gate.** Check the deliverable with a command or grader you control. A child's "all tests
   pass" counts its own tests, not yours. For a diagnosis or a risky change, give the same
   evidence to a second child with the opposite prior and compare.
6. **Report** what was checked and how; say what was not.

## Routing

| Tier | Use for | Model |
|---|---|---|
| Fast | Scouting, log and data queries, edits with a complete spec | Cheapest model the harness offers |
| Skeptical | Review, diagnosis, any brief that may itself be wrong | A stronger model than the author's |

Default subagent model: `mistral/deepseek-v4-1-flash-1017` at low thinking (owner directive,
2026-11-15). Low is the floor for lookups and mechanical edits; reserve `max` for a one-off task that
needs it, such as authoring a skill. The tiers apply when the harness offers cheaper models.

Thinking level by task (owner directive, 2026-11-15): low for trivial or mechanical work
(lookups, small edits, summaries, log queries); high for hard problems (multi-file fixes,
diagnosis, streaming or security work); max for high-stakes decisions (commit or run strategy,
evidence gates, security release). Low stays the floor; escalate only when the task warrants it.

## What worked

- One owner per file per wave; the owned-file list goes in the brief.
- One shared common-rules brief plus one task brief per child.
- Scouts write a report file and return at most 150 words.
- The orchestrator re-runs the tests and adds a hidden check before accepting.
- Red-team children demonstrate by execution and work on copies.
- A child that finds the brief wrong reports it; three brief errors were caught this way.

- Fast-tier models execute a brief as written, including its mistakes. Skeptical-tier models
  report a flawed brief instead of implementing it silently.
- Raise the thinking level only after a failed check; higher levels cost time and tokens
  without a default gain.
- Above these tiers, do the work yourself or use a top-tier model. Do not assume the table
  holds for tasks longer than about 10 minutes, large repositories, hardware or vision.

## Traps

- **Sandboxed harnesses.** Some deny paths outside the project. Put fixtures and briefs inside
  an allowed path before launching, and treat "access denied" as a brief error, not a model
  failure.
- **Approval modes.** A child in an interactive permission mode waits forever. Set the
  non-interactive mode when launching.
- **Spawn cost.** A fresh child pays for its whole prompt once. Run a one-command question
  yourself.

## Keeping the table honest

Re-measure when a model is added or the table looks stale: same fixtures, hidden checks and
known answers; record correctness, wall time and whether a planted brief flaw was flagged.
If every candidate passes, the task is too easy to rank; add a harder one before rerouting.
