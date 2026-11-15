# Compression program: decided and open

Pre-registered plan for `studies/program-v1/`. Not evidence: no number here comes from a run.
The protocol, phases and gates are frozen in [protocol.json](../studies/program-v1/protocol.json);
the evidence namespace is [studies/program-v1/README.md](../studies/program-v1/README.md).

## Decided

### Precision surface

Fix everything except the target module. Keep the world model resident, keep embeddings BF16, and
quantize only the target linear layers, so precision is isolated from deployment changes.

| Curve | Points | Trials |
|---|---|---|
| Language | INT8, NF4, HQQ3, HQQ2 | 100 development |
| Vision | INT8, NF4, HQQ3, HQQ2 | 100 development |
| Action | INT8, NF4, HQQ3, HQQ2 | 100 development |
| Joint vision+action | Vision HQQ2 + action HQQ2, language BF16 | 100 development |

One pre-registered variant per module; every tested variant stays in the study record. Do not take
the best variant per width, which biases the curve upward.

Plot success against effective bits, not nominal bits:

\[
b_{\mathrm{eff}}(x) = \frac{8 \, S(x)}{N(x)},
\]

where \(S(x)\) is resident tensor storage for the module, including scales, zeros and metadata, and
\(N(x)\) is the logical parameter count. Nominal 3-bit HQQ can be worse than 4-bit NF4 because of
axis and group size.

Primary comparisons, pre-registered:

- H1: language 2-bit loses more than 10 pp against B16.
- H2: vision 2-bit is non-inferior within 5 pp.
- H3: action 2-bit is non-inferior within 5 pp.
- H4: joint vision+action 2-bit is non-inferior within 5 pp; the interaction is not superadditive.

### Format curve

Decoder-only, same scope, 100 development trials for NF4, INT8, HQQ4, HQQ3, HQQ2, GPTQ and the
AWQ-style diagonal adaptation. Plot success against peak memory and effective bits. This curve
separates a bit-width effect from a format effect: group size, axis and calibration.

### Execution ladder

Memory, stock latency, graph latency, dense versus packed NF4, one versus four flow steps. The
CUDA-graph engine is an independent execution intervention; Dream-RSI did not discover it. Label
any one-flow-step B16 graph number as a timing-only control unless its success was measured.

### Offline screening

Action change on fixed frames is a rejection test, not a ranking function: it separates broken
recipes from working ones and cannot order near-lossless ones. Near-lossless recipes need
closed-loop trials. Report the threshold used and the correlation per action-MSE bin.

### Analysis rules

- Wilson 95% interval per arm; task-stratified paired bootstrap against B16 on identical states.
- Report measured points with intervals. Do not smooth them; a model may choose the next
  experiment, never replace a measurement.
- Reserved trials never guide selection. No phase is pooled with another.
- Holm correction over the four primary comparisons; every other comparison is exploratory.
- One context per front: phase, hardware, flow steps and calibration fixed. A candidate may not
  choose the context in which it wins.
- The joint front is the Pareto set of the product objective, not the union of per-objective
  fronts. Individual fronts are diagnostic slices.
- Replay value \(V\) ranks exploration policies over frozen discovery trees. It is not LIBERO
  success and is reported as a separate judgment.

### Stop rules

- A 100-trial point with a Wilson interval wider than ±10 pp cannot support a non-inferiority
  claim. Run 500 or drop the claim.
- A recipe that fails smoke at 6/10 or worse stops; do not promote.
- A format with screen action MSE above \(10^{-3}\) and smoke below 8/10 stops.
- If replay value improves but development success does not, report the negative result.
- If GPU time slips, drop the format figure to a table before dropping the precision curve.

## Open

- **Runs.** No smoke or development trial has been launched. The precision surface is
  pre-registered in [precision_surface_plan.json](../studies/program-v1/precision_surface_plan.json)
  with 13 recipe arms under `recipes/`.
- **Measurement queue.** `evaluation/compression/queue.py` runs the pre-registered surface on one
  GPU slot; it is pending merge, so `vla-compress run` is the current path for one arm and phase.
- **Dream-RSI clean loop.** Discovery agents write `proposal.md` and `solution.py`; the fixed
  evaluator scores them; replay selects the exploration policy. No evidence run yet. Fidelity and
  deviations: [DREAM_RSI.md](DREAM_RSI.md).
- **Reserved confirmation.** Freeze one full-vocabulary recipe before running 500 reserved trials.
  Report the paired difference and Wilson interval; claim non-inferiority only if the interval stays
  above −5 pp. Do not re-select after seeing reserved outcomes.
- **Parallel rollouts.** `evaluation/parallel_rollout.py` is not wired to arbitrary compression
  recipes. Before using it for evidence: serial-versus-parallel equivalence at the same seeds and
  states, mode-consistent comparisons, reserved confirmation on the serial path.
- **Report.** `report/main.tex` is a scaffold; `report/plots` skips figures whose evidence is absent.

## Milestones

| Date | Deliverable |
|---|---|
| Nov 17-20 | Smoke screens, then 100-trial development for language, vision, action and the joint 2-bit cell |
| Nov 23-24 | Format curve |
| Nov 25-29 | One Dream-RSI clean cycle on the precision axes; analysis with Wilson intervals and the paired bootstrap |
| Nov 30 | Freeze claims; at most two reserved arms |
| Dec 1-2 | Rebuild `report/plots`, `report/figures`, `report/data/numbers.json`; enforce the 5-page limit |
| Dec 3-4 | Presentation |
| Dec 9 | Report due |

## Out of scope

No Qwen training. No reserved trial in selection. No claim that replay value is LIBERO success. No
claim that an audit tends to approve.
