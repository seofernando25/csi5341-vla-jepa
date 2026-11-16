# Roadmap: from here to the final report

Written 2026-11-16 by the orchestrating agent for Fernando and Noah. It lists everything left
before the presentation (2026-12-04) and the report (2026-12-09). Live run state is in
`HANDOFF.md`; this file is the plan. Tick a box when its acceptance check passes, and delete this
file when the report is submitted.

## 1. Requirements

What the report has to show. Each requirement names its source.

| ID | Requirement | Source | Status |
|---|---|---|---|
| R1 | Decoder-only INT8 and NF4 against the BF16 reference, 500 matched episodes each, with success, peak memory, latency and intervals | Proposal | Done |
| R2 | Verdict on the hypothesis: at least 25 % less peak memory, success within 5 points | Proposal | Done (NF4 36.9 % passes, INT8 23.3 % misses) |
| R3 | Failure analysis on identical observations: representations, spatial attention, action trajectories, on the tasks that drop | Proposal | Partial: frames done, episode trajectories missing |
| R4 | Dream-RSI used to search for structural changes that keep performance | Proposal ("if time permits") | Done for size; latency pending |
| R5 | Every number traceable to `report/data/numbers.json`, built by `vla-report-plots` from `studies/` | `CLAUDE.md` | Holds today; recheck after each evidence change |
| R6 | IEEE two-column, at most 5 pages | Course | 4 pages now |
| R7 | Plain "we" voice, few numbers in prose, no AI tell-tale phrasing | Owners | Draft passes one review; recheck after new sections |
| R8 | Reserved episodes never guide selection; phases never pooled | `CLAUDE.md` | Holds; keep for v3 and v4 |
| R9 | Credit for each author's work, including Noah's Dream-RSI integration | Owners | Acknowledgment restored; Noah to confirm wording |
| R10 | A search on latency, the owners' "second step": fuse and speed up the policy while it keeps working | Owners, 2026-11-16 | Prepared, not run |

## 2. Design

### 2.1 Story of the paper

1. VLA models are spreading and carry large language backbones. VLA-JEPA needs 5.6 GB.
2. A smaller backbone failed (11/500), so we compress the released weights.
3. Precision: the language decoder is the fragile part (breaks between 3 and 2 bits); NF4 meets
   the proposal's target, INT8 narrowly misses it.
4. Why: the decoder's image tokens drift first; the policy holds until the drift reaches the
   action head (failure analysis).
5. Execution: the CUDA graph matters more for speed than any weight format.
6. Search: Dream-RSI removes 83 % of memory with no measurable loss, by removing modules first.
   A debiased rerun (v3) shows what it finds without our code and the prompt tokens.
7. The search gamed our memory score; we show the exploit and the fix.
8. Latency search (v4): what the same loop finds when the goal is speed.
9. What to use; limitations.

### 2.2 Figure plan (target: at most 7 figures in 5 pages)

| Figure | Shows | Status |
|---|---|---|
| Proposal arms | BF16, INT8, NF4 side by side: reserved success with interval, peak memory, latency | To build |
| Precision by bits | Development success by weight bits and module | Done (`figures/precision.pdf`) |
| Failure analysis | CKA and action error by bits on 80 frames | Done (`figures/failure.pdf`); add trajectory panel if R3 lands |
| Execution ladder | Peak memory and latency per execution arm | Done; relabel internal names (Q8, B16 graph) |
| Reserved results | Change in reserved success by peak memory, all eight reserved cells | Done (`figures/reserved.pdf`); add v3 and v4 winners |
| Size timeline | Best memory removed over probes for v1, v2, v3, each step labelled by what the agent did | To build after v3 |
| Latency timeline | Best speedup over probes for v4, steps labelled | To build after v4 |
| Dreaming | Replay score on seen and held-out trees per cycle, v1 to v3 | To build; may merge into the size timeline |
| Action error vs success | Offline screen does not rank | Done; candidate to cut for space |

Rules for every figure: single column unless the data needs width; no statistics text inside the
plot (it goes in the caption); readable at 3.5 in; axis range set by the quantity (a difference may
use a range around zero; a success rate starts at 0).

### 2.3 Latency search scoring (v4)

Score is the lower bound of the 95 % interval of the speedup over the reference, if the smoke gate
passes, peak memory does not exceed the reference's, and actions stay close to B16 on unseen and
perturbed frames. Timing alternates reference and candidate blocks in one evaluation. Guards cover
clock tampering, cached answers, background work and work moved off the GPU. Remaining holes are
listed in `studies/program-v4/README.md` on its branch.

## 3. Tasks

Owner F = Fernando's agent, N = Noah's agent, H = a human decision. GPU tasks run one at a time.

### 3.1 Experiments

- [ ] **T1. Finish the debiased size search (v3).** Running since 07:22, ends by about 15:20.
  Acceptance: three cycles recorded, no node read the hidden code or used token ids (watcher
  report), evidence committed with relative paths. Owner F.
- [ ] **T2. Select v3 winners on development data, freeze, run Reserved-500.** Acceptance: freeze
  record committed before the first reserved episode; reserved results in
  `studies/program-v3/reserved/`. About 1 h per cell. Owner F.
- [ ] **T3. Close the latency-score holes (Q2).** Acceptance: four identity runs score at most 1.0
  with point estimates within 5 %; a nearest-neighbour cache fails a test; checks pass. Owner F.
- [ ] **T4. Run the latency search (v4).** Needs the GPU free of other tenants (ask Fernando to
  close `llama-server`). About 8 h. Acceptance: three cycles, every winner re-timed on an idle GPU
  with `vla-core-benchmark`, levers used by each winner listed. Owner F, decision H.
- [ ] **T5. Reserved-500 for the v4 winners**, with latency re-measured on an idle GPU. Owner F.
- [ ] **T6. Episode-level failure analysis (R3).** Replay identical episodes for B16 and the
  2-bit language cell on tasks 1, 4 and 9, and for INT8 on task 5; record where the arm
  trajectory departs and the outcome. About 30 GPU-minutes. Acceptance: compact trajectory record
  under `studies/program-v1/failure/` and one figure panel. Owner F.

### 3.2 Figures

- [ ] **T7. Proposal-arms figure** (BF16, INT8, NF4). Owner F.
- [ ] **T8. Size timeline** for v1, v2, v3 with labelled steps, in the style of a best-so-far
  progress curve. After T1. Owner F.
- [ ] **T9. Latency timeline** for v4. After T4. Owner F.
- [ ] **T10. Dreaming panel** (seen and held-out replay score per cycle). Owner F.
- [ ] **T11. Ladder relabel**: plain names, no "Q8" or "timing" tags. Owner F.
- [ ] **T12. Precision figure**: separate the crowded "vision" and "language" end labels. Owner F.
- [ ] **T13. Reserved figure**: add v3 and v4 winners after T2 and T5. Owner F.

### 3.3 Report text

- [ ] **T14. Add v3 results** to the search section: what the search found without our hints,
  and whether the direction (remove modules first, then quantize) still appears. Owner F.
- [ ] **T15. Add the latency search** as its own subsection, with its exploit guards in one
  paragraph. Owner F.
- [ ] **T16. Trajectory paragraph** in "Why 2 bits breaks" after T6. Owner F.
- [ ] **T17. Voice pass** on every new paragraph (Haiku, then line-by-line review); no number or
  claim changes. Owner F.
- [ ] **T18. Page budget**: stay at 5 pages or fewer; cut the action-error figure first if needed.
- [ ] **T19. Acknowledgment wording** confirmed by Noah. Owner H (Noah).
- [ ] **T20. Final read** of the PDF by both authors. Owner H.

### 3.4 Codebase and docs cleanup

- [ ] **T21. Merge branches**: `t3/implement-brief-q` (latency study) after T4, the v3 worktree
  branch after T1; remove the worktrees and delete the branches the same day (`CLAUDE.md`
  "Branches"). Owner F.
- [ ] **T22. Correct `HANDOFF.md`**: its 05:15 table swaps the two v1 winners (cycle 1 scored
  93.4 %, cycle 2 95.6 %). Owner F.
- [ ] **T23. Retire the unused format figure** (`fig_format` still builds `figures/format.pdf`,
  which the report no longer includes). Owner F.
- [ ] **T24. Study READMEs**: v3 and v4 each get question, result, evidence and reproduction
  command; `studies/README.md` indexes them. Owner F.
- [ ] **T25. `docs/DREAM_RSI.md`**: add the v3 exclusions and the v4 score as protocol choices. Owner F.
- [ ] **T26. Presentation** (`web/presentation/`) still tells the pre-clean-room story; update it
  from the final figures before 2026-12-04. Owner F and N.
- [ ] **T27. Push and board record** only after Fernando says so. Owner H.

## 4. Order and timing

| When | GPU | Other work |
|---|---|---|
| Now to about 15:20 | v3 search (T1) | T3, T7, T10, T11, T12, T22, T23 |
| After T1 | T6 (30 min), then T2 (about 2 h) | T8, T14, T24 |
| After T2, GPU idle | T4 (about 8 h) | T25, T26 draft |
| After T4 | T5 | T9, T13, T15, T16, T17 |
| Before 2026-12-04 | none | T26, T18, T20 |
| Before 2026-12-09 | none | T19, T20, T21, T27 |

## 5. Risks

- **Shared GPU.** Other processes (`llama-server`, notebooks) slow runs and corrupt latency
  timing. Run T4 only on an idle GPU.
- **Another exploit.** The latency score is easier to game than memory. Read every v4 winner's
  code before it is reported.
- **v3 may find less.** Without our code and the prompt tokens it may not reach 83 %. That is a
  result: report it as the cost of the hints.
- **Page limit.** Seven figures in five pages is tight; merge timelines before cutting text.
