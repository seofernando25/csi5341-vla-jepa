---
name: terse-writing
description: Write and edit prose concisely. Use for the report, READMEs, docstrings, comments, commit messages, board records and replies to the user, and when asked to tighten existing text.
---

# Terse writing

Every sentence must carry a fact, a decision or an instruction. Delete the rest.

## Rules

1. Lead with the result or the blocker. Context comes after, only if needed.
2. One claim per sentence. Active voice, concrete verbs, present tense for code behavior.
3. Numbers carry their basis: trial count, unit, interval and comparison.
   "Q4: 479/500 (95.8%), paired change +0.8 pp [−0.6, +2.2] vs B16", not "Q4 performs well".
4. Name the thing: file path, function, configuration ID (B16, Q8, Q4, C1–C3), run ID or commit.
5. State uncertainty once, precisely: what was not tested and why it matters.
6. Prefer a table or list over a paragraph when items are parallel. No list for one item.
7. Use one term per concept. "Peak allocated memory" is not "VRAM" in the next sentence.
8. Units: GB means 10^9 bytes, GiB means 2^30. Never mix them in one table.

## Cut on sight

- Hype: crucial, robust, comprehensive, powerful, seamless, novel, state-of-the-art, deep dive.
- Filler: "it is important to note", "in order to", "essentially", "simply", "as mentioned above".
- Self-narration: "I will now...", "Great question", closing summaries that repeat the body.
- Stacked hedges ("may potentially suggest"). Pick one verb.
- Decoration: emoji, bold on every other phrase, headers on a three-line answer.
- Process history in user-facing docs. Git and the board keep history; docs state the current state.

## By medium

- **Reply to the user**: answer first. Then what changed, what was verified (command and outcome), what remains.
- **README and study docs**: purpose in one sentence, then results or procedure. Link to evidence instead of restating it. Delete outdated sections; do not append corrections.
- **Report**: claim, number with interval, figure or table reference. Limitations once, in their own section.
- **Docstrings and comments**: the contract (inputs, outputs, units, invariants) or the non-obvious reason.
- **Commit messages**: imperative subject ≤72 characters; the body says why and how it was verified.
- **Board records**: one line: result, meaning, caveat, next step.

## Example

Before:
> The GPU optimization introduced a real input defect. Structured processor arguments silently
> overrode the intended no-rescale setting, and it is important to note that images were divided
> by 255 again, which is a crucial finding.

After:
> The GPU processor rescaled 0–1 images a second time (255× lower contrast). `smolvlm_interface.py`
> now passes `do_rescale=False`; a pixel-range regression test covers it.

## Editing existing text

Tighten in place, preserving every fact, number and link. If facts conflict with the code or newer
results, fix the text to match the evidence and say so in the commit message. Do not change
scientific claims while editing for style.
