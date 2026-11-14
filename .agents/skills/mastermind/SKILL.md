---
name: mastermind
description: Fan a job out to cheap subagents and gate what comes back. Use when the user says /mastermind, asks to delegate, parallelise or "use subagents", or when a task splits into independent pieces.
---

# Mastermind

The orchestrator plans, briefs, verifies and decides. Subagents do the reading, computing and
editing.

## Loop

1. **Split.** One deliverable per child, checkable without reading the child's transcript: a
   file, a JSON answer, a passing command. Pieces that touch the same file or the same
   exclusive resource (a device, a GPU, a deploy target) are one piece with one owner.
2. **Route** each piece by tier (below). Start at the cheapest tier that fits; escalate only
   after a failed check.
3. **Brief.** Write the brief to a file and tell the child to read it. State the working
   directory, what it may and may not touch, the exact output file and format, how it must
   verify itself, and a report length cap. Give paths and line ranges, not "find the code".
4. **Launch async**, all independent children in one message. Do not poll.
5. **Gate.** Check the deliverable with a command or grader you control. A child's "all tests
   pass" counts its own tests, not yours. For a diagnosis or a risky change, give the same
   evidence to a second child with the opposite prior and compare.
6. **Report** what was checked and how; say what was not.

## Routing

| Tier | Use for | Models |
|---|---|---|
| Fast | Scouting, log and data queries, edits with a complete spec | Beam free, Inkling 2 Small free |
| Skeptical | Review, diagnosis, any brief that may itself be wrong | Gemini 4 Hydrogen (low), Haiku 5.5, Iceman free |

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
- **Token counts differ per harness.** Compare wall time and outcomes across harnesses, not
  reported tokens.

## Keeping the table honest

Re-measure when a model is added or the table looks stale: give every candidate the same
fixtures, grade with hidden checks and known answers, and record correctness, wall time and
whether a planted flaw in the brief was flagged. If every candidate passes, the task is too
easy to rank them; add a harder one before changing the routing.
