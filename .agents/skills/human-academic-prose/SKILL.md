---
name: human-academic-prose
description: Write and revise research papers, project reports, abstracts, and reader-facing essays with clear, natural human voice, strong narrative flow, and rigorous technical accuracy. Use when drafting or humanizing paper prose, eliminating number dumps or robotic phrasing, or polishing LaTeX reports for submission.
---

# Human academic prose

Write like an engineer or researcher explaining real findings to a sharp colleague: clear, grounded,
specific, and pleasant to read. Precision does not require sounding like a machine log, and polish
does not require hype.

Use `terse-writing` for commit messages, board entries, docstrings, and terse status updates. Use
this skill for papers, reports, abstracts, and any long-form prose meant to be read straight
through.

## Core principles

1. **Carry the argument in prose; put the ledger in tables and figures.**
   Avoid "number diarrhea" (stacking three fractions, percentages, and confidence intervals into a
   single sentence). Keep one or two anchor numbers in the text to drive the point, round sensibly
   in narrative (e.g., `5.6 GB`, `73 ms`, `95%`), and direct the reader to the table or plot for
   exact decimals, per-row counts, and confidence intervals.
2. **Vary sentence rhythm naturally.**
   Avoid monotonous staccato fragments ("X did Y. Z did W. A held. B failed.") and avoid bloated
   four-clause academic fog. Pair a crisp claim with a fuller sentence that explains the mechanism
   or cause.
3. **Explain the mechanism, not only the score.**
   Whenever a result surprises or breaks, state *why* in concrete terms (for example, visual tokens
   drifting inside a quantized decoder before the distortion reaches the action head) rather than
   reciting metrics alone.
4. **Keep caveats calm and honest.**
   Do not oversell a result or hide a failed run, timeout, overfitting meta-agent, or evaluator
   loophole. State each limit once, plainly, where it belongs, without hedging every verb.
5. **Preserve technical accuracy above style.**
   Never distort a unit, denominator, bit width, or scope boundary to make a sentence smoother.
   Check every number against its source record (`report/data/numbers.json` or the original paper)
   after editing.

## Ban list (cut on sight)

- **Em dashes (`---` or `—`).** Use commas, colons, parentheses, semicolons, or a second sentence.
- **Synthetic antithesis templates.** Cut "Not X. Y.", "It is not about X; it is about Y.", and
  theatrical contrasts used as section hooks.
- **Dead metaphors and purple prose.** Cut "Achilles' heel", "heavy lifting", "pries open",
  "pay dividends", "bats 100%", "quietly", "delve", "tapestry", "landscape", "testament",
  "paradigm", "myriad", "nuanced".
- **Stiff LLM transitions and hype.** Cut "Furthermore", "Moreover", "Taken together", "Crucially",
  "Notably", "Importantly", "In essence", "Put simply", "It is important to note", "beyond doubt",
  "robust", "seamless", "pivotal", "leverage", "underscore", "comprehensive", "compelling".
- **Robotic signposting.** Never open a section with "In this section we will discuss..." or close
  a paper with "In conclusion, we have shown...". Lead with the finding.

## Pre-submission checklist

1. **Fact and unit audit:** Verify every number, unit (`k` vs. million vs. billion, `GB` vs.
   `GiB`, nominal vs. effective bits), and cross-reference against primary data files.
2. **Consistency check:** Ensure the abstract, body, tables, captions, and conclusion agree on
   every qualification (e.g., do not claim a factor was untested in Section 4 and then call it
   "for little return" in Section 5).
3. **Telltale scan:** Grep the LaTeX source for em dashes and banned phrases before compiling.
4. **Layout and page balance:** Compile to PDF, verify zero LaTeX errors or undefined references,
   and check per-page word density so the paper hits its exact page target without an orphan tail
   on the last page.
