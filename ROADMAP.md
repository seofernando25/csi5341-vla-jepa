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
| R4 | Dream-RSI used to search for structural changes that keep performance | Proposal ("if time permits") | Done for size (v1, v2); latency next; debiased size rerun paused |
| R5 | Every number traceable to `report/data/numbers.json`, built by `vla-report-plots` from `studies/` | `CLAUDE.md` | Holds today; recheck after each evidence change |
| R6 | IEEE two-column, at most 5 pages | Course | 4 pages now |
| R7 | Plain "we" voice, few numbers in prose, no AI tell-tale phrasing | Owners | Draft passes one review; recheck after new sections |
| R8 | Reserved episodes never guide selection; phases never pooled | `CLAUDE.md` | Holds; keep for v3 and v4 |
| R9 | Credit for each author's work, including Noah's Dream-RSI integration | Owners | Acknowledgment restored; Noah to confirm wording |
| R10 | A search on latency, the owners' "second step": fuse and speed up the policy while it keeps working | Owners, 2026-11-16 | Running (v4); first nodes about 2.7x |

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
perturbed frames. Timing alternates reference and candidate blocks in one evaluation, and every task
is timed equally. Guards cover clock tampering, cached answers, background work and work moved off
the GPU. Remaining holes are listed in `studies/program-v4/README.md`.

## 3. Tasks

Owner F = Fernando's agent, N = Noah's agent, H = a human decision. GPU tasks run one at a time.
Each task says why it matters, the steps, the traps, and the check that closes it.

### 3.1 Experiments

#### T3. Close the latency-score holes (in progress)
*Why.* A score that a search optimises gets attacked. The first latency evaluator let the untouched
policy "beat itself" by up to 30 % on a shared GPU, and a table of cached actions could have passed.
Either would produce fake wins.
1. Time the reference and the candidate in alternating blocks inside one evaluation, so a burst
   from another GPU process hits both equally.
2. Report the speedup with a 95 % interval over blocks and score its lower bound, so noise can only
   lower a score, never raise it.
3. Record what other processes do on the GPU while timing, and flag the node if they were busy.
4. Check that a candidate still acts like B16: on frames it has never seen, and on slightly
   perturbed copies of known frames, its action error against B16 must stay under a threshold set
   from the screen records, and its actions must move when the input moves.
5. Tests: a nearest-neighbour cache fails; the identity and an NF4 solution pass.
*Traps.* Setting the threshold by feel instead of from recorded screens; testing only on the frames
the agent can see.
*Done when* four identity runs score at most 1.0 with point estimates within 5 %, the cache test
fails, and all checks pass.

#### T4. Run the latency search (v4)
*Why.* The owners' second step: let the loop find a faster policy that still works, the way the
size search found a smaller one.
1. Review the latency code before merging it: split files that do two jobs, merge duplicated
   helpers, delete leftovers, one `vla-*` command row per runner (T21 covers the merge).
2. Merge into `main`, then launch from a fresh worktree of `main` so `main` stays editable.
3. Make sure the GPU has no other tenant (no `llama-server`, notebooks idle).
4. Launch with one GPU slot. Timing must never share the card with another evaluation.
5. The first node is the untouched policy. It must score about 1.0 or below; above 1.05 means the
   score is biased, so stop and fix.
6. A watchdog checks the run every 20 minutes: guard violations, cached answers, timer tricks,
   speedups that come from fewer flow steps or a smaller model (allowed, but reported as such), and
   any speedup above 3×, compared with our hand-built CUDA graph (22 ms at 4 flow steps).
*Traps.* Comparing a 1-flow-step candidate with a 4-step reference without saying so (we made this
mistake in the draft once); trusting timings taken while the GPU was shared.
*Done when* three cycles are recorded and every winner is re-timed on an idle GPU with
`vla-core-benchmark`.

#### T5. Reserved-500 for the latency winners
1. Pick winners on development data only and write a freeze record before any reserved episode.
2. Run 500 reserved episodes per winner; re-measure latency on an idle GPU.
3. List the levers each winner used (graph capture, fusion, custom kernels, fewer flow steps).
*Done when* the results and the freeze record are committed.
*Status.* Done: [program-v4](studies/program-v4/README.md) selection and reserved 500.

#### T1. Resume the debiased size search (v3, paused)
*Why.* The first two searches could read our compression code and were handed the 53 prompt
token ids, and most nodes used both. v3 hides both, so it shows what the search finds on its own.
1. Resume after T4 and T5 with the same `vla-rsi start` command from the v3 worktree; it continues
   from `loop_state.json` and reruns the node that was in progress when it was paused.
2. Restart a watchdog that also flags any node that reaches the hidden code or uses token ids.
3. Commit the evidence with relative paths.
*Traps.* Comparing v3's best memory with v1's without saying v1 had the hints.
*Done when* three cycles are recorded and the watchdog reports no leak.
*Status.* Done: three cycles, [program-v3](studies/program-v3/README.md). Nodes still hardcode or
re-derive the prompt token ids.

#### T2. Reserved-500 for the v3 winners
Same steps as T5, with memory measured under the fixed rule.
*Status.* Done: reserved 500 for each cycle's best node, in [program-v3](studies/program-v3/README.md).

#### T6. Episode-level failure analysis
*Why.* The proposal promised analysis on "identical observation sequences" and "downstream action
trajectories". We have it on 80 single frames; we do not yet have whole episodes.
1. Replay the same episodes (same state and seed) for B16 and the 2-bit language cell on tasks 1,
   4 and 9, where the 2-bit cell fails every time, and for INT8 on task 5, where it loses two.
2. Record the arm trajectory per step and the gripper state for both policies.
3. Find the first step where the trajectories part by more than a set distance, and what happens
   after (missed grasp, wrong object, timeout).
4. One small figure panel: distance between trajectories over time, one line per task.
*Traps.* Different seeds between the two policies make any divergence meaningless; check the first
observation is identical.
*Done when* a compact record is under `studies/program-v1/failure/` and the panel builds.

### 3.2 Figures

Rules for all: one column unless the data needs width; statistics in the caption, not the plot;
readable at 3.5 in; a success rate starts at 0, a difference may centre on 0.

- **T7. Proposal arms.** BF16, INT8 and NF4 side by side: reserved success with interval, peak
  memory, latency. This answers the proposal's question in one picture.
- **T8. Size timeline.** Done for v1 and v2 (`figures/size-timeline.pdf`, `fig:size-timeline`); add
  v3 after T1. Best memory removed after each probe for v1, v2 and v3. Label each jump with
  what the agent did ("drops the world model", "53-row embedding", "3-bit packing"). Mark the
  exploit nodes as invalid, not as records. After T1.
- **T9. Latency timeline.** Done for the partial v4 snapshot (`figures/latency-timeline.pdf`,
  `fig:latency-timeline`); redraw with final v4 numbers after T4. Same picture for v4: best speedup
  after each probe, jumps labelled.
- **T10. Dreaming panel.** Replay score on seen trees and on held-out trees per cycle. It shows the
  policy fitting its own history while held-out value stays flat.
- **T11. Ladder relabel.** Plain names ("INT8 decoder", "CUDA graph, 4 steps"); drop "Q8", "B16" and
  "(timing)".
- **T12. Precision labels.** Separate the crowded "vision" and "language" end labels.
- **T13. Reserved figure.** Add the v3 and v4 winners.

### 3.3 Report text

- **T14. v3 in the search section.** Done: the third search, with the code hidden, reached 82.8 % and 480/500
  reserved, about what the hinted first search found.
- **T15. Latency subsection.** Done: reserved and idle-GPU timing for the three v4 winners.
- **T16. Trajectory paragraph** in "Why 2 bits breaks", after T6.
- **T17. Voice pass.** Done 2026-11-17: Haiku wrote v3 in, then a separate Haiku review checked
  numbers against the evidence, plain language and every figure. Recheck after any new paragraph.
- **T18. Page budget.** Done: 5 pages, no overfull boxes.
- **T19. Acknowledgment.** Noah confirms the sentence about his work.
- **T20. Final read.** Both authors read the PDF once, aloud if possible.

### 3.4 Codebase and docs cleanup

- **T21. Branches.** Done: the v3 and v4 runs ended and their evidence is on `main`; the v3
  worktree and branch are removed (tag `program-v3-debiased` stays).
- **T22. Handoff.** Done: the swapped v1 winner results are corrected.
- **T23. Format figure.** Done: `fig_format` is retired and the report does not use it.
- **T24. Study READMEs.** Done: `studies/program-v3/README.md` and `studies/program-v4/README.md`
  give question, result so far, evidence and reproduction command, and are indexed in
  `studies/README.md`. Refresh the results when each run ends.
- **T25. `docs/DREAM_RSI.md`.** Done: section 5 records the v3 exclusions and the v4 score.
- **T26. Presentation.** Open: `web/presentation/` still tells the pre-clean-room story; it waits
  for the final figures, then is rebuilt before 2026-12-04.
- **T27. Push and board record.** Open: only after Fernando says so.

## 4. Order and timing

| Step | GPU | Alongside |
|---|---|---|
| 1 | T3 checks (short) | T7, T10, T11, T12, T23 |
| 2 | T4 latency search (about 8 h) | T6 analysis code, T24 and T25 drafts |
| 3 | T5 reserved for v4 winners | T9, T15 |
| 4 | T1 resume v3 (about 8 h) | T26 draft |
| 5 | T2 reserved for v3 winners, then T6 (30 min) | T8, T13, T14, T16 |
| Before 2026-12-04 | none | T26, T17, T18 |
| Before 2026-12-09 | none | T19, T20, T21, T27 |

## 5. Risks

- **Shared GPU.** Other processes (`llama-server`, notebooks) slow runs and corrupt latency
  timing. Run T4 only on an idle GPU.
- **Another exploit.** The latency score is easier to game than memory. Read every v4 winner's
  code before it is reported.
- **v3 may find less.** Without our code and the prompt tokens it may not reach 83 %. That is a
  result: report it as the cost of the hints.
- **Page limit.** Seven figures in five pages is tight; merge timelines before cutting text.
- **Invalid v1 nodes.** v1 cycle-3 nodes `iter_002-b1-a3`, `iter_002-b2-a1` and `iter_002-b2-a2`
  are the memory exploit; never report or plot them. The stopped v4 first launch is the second
  exploit and is never selected.
- **Weak gates.** Smoke is 10 episodes and the untouched policy scores 8/10, exactly the gate. v1
  policy selection is in-sample only and overfit.
- **Calibration tie.** The embedding shrink some nodes use ties the policy to the calibration
  prompts; say so in the report.
- **Untested paths.** The open policy agent passed one synthetic step before launch, not a live
  cycle; the anytime-salvage path has CPU tests only.

