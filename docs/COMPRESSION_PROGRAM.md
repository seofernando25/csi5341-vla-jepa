# Compression Program: From Precision Curves to a Context-Conditioned Pareto Search

Status: plan, 2026-11-15. Not evidence. No GPU run was launched to write this.

## 1. Purpose

The report currently makes four kinds of claims: module precision sensitivity, memory/latency
effects, joint-recipe quality, and replay-guided search. The first three are measured; the fourth is
development-only. The claims are stronger than the evidence in two places: the module sweep is a
10-trial pilot, and the Dream-RSI result is not held-out confirmed.

This plan does three things:

1. Adds the missing precision response surfaces, with enough trials to support a curve.
2. Formalizes the proposed context-conditioned Pareto program, including its limits.
3. Separates what the Dream-RSI harness can already do from what must be built for a faithful
   meta-search overseer.

## 2. Claim Audit

| Report claim | Current evidence | Verdict | Action |
|---|---|---|---|
| Vision and action tolerate 2-bit weights | 10-trial smoke, 8/10 each; B16 8/10; Wilson 95% [49, 94] | Pilot only; not a curve; joint vision+action not tested | Run 100-trial development curves and a joint 2-bit cell |
| Language decoder is the precision bottleneck | 2-bit 1/10 (HQQ axis 0), 0/10 (axis 1); 3-bit 7/10 (axis 0), 0/10 (axis 1); 4-bit 7/10 (NF4) | Directional, but confounded with quantizer geometry and 10-trial noise | Separate bit width from format/axis; 100-trial language curve |
| Decoder-only NF4 cuts peak memory 37% with no detectable loss | Q4 479/500 = 95.8%; paired +0.8 pp [-0.6, +2.2]; peak 3.54 GB | Supported as non-inferiority within 5 pp, not as exact equivalence | Reword to "non-inferior within the 5-point margin" |
| Two joint recipes under 1.6 GB at ≥96% success | C1 484/500 = 96.8%, 1.38 GB; C2 490/500 = 98.0%, 1.51 GB | Supported; C1 gain not significant; C2 gain concentrated in task 5 | Keep; report task concentration and unexplained cause |
| Compression alone slows the policy | Q4 93.6 ms vs B16 77.5 ms; C1 142.9 ms; stock path launch-bound | Supported; launch-bound explanation is an inference from kernel/sync counts | Keep; label the mechanism as inferred |
| CUDA-graph engine removes overhead; 31.6 ms vs 24.8 ms | C2 31.6 ms; B16 graph 1-step 24.8 ms, timing-only control | Supported at matched flow steps | Keep; label 24.8 ms as timing-only, not a success arm |
| Offline action change screens broken recipes but cannot rank near-lossless ones | 163 screened recipes: overall Spearman ρ = -0.67; MSE ≥ 1e-3 ρ = -0.82; MSE < 1e-3 ρ = -0.25; MSE < 1e-4 ρ = +0.19 (p = 0.13); MSE < 1e-5 ρ = 0.01 | Supported; the sentence is correct but opaque | Explain the threshold and report the bin correlations |
| Dream-RSI trims weights further on development trials | Stage 2 85/100 at 904.8 MB; R1 83/100 at 853.1 MB; R3 88/100 at 885.1 MB; R4 86/100 at 1199.6 MB; R5 85/100 at 876.0 MB | Development-only; not held-out confirmed; "further" needs a reference | Say "finds a full-vocabulary recipe 89 MB smaller at matched development success"; label search-only |
| Replacing the backbone fails | SmolVLM2-500M 2.2% on 500 held-out | Supported negative result | Keep as a separate boundary result, not in the same sentence as Dream-RSI |

The biggest correction: the module sweep is a screen, not a response curve. "Best tested variant at
each width" also takes a maximum over variants, which biases the curve upward. Replace that with one
pre-registered variant per module and report all tested variants in the study record.

## 3. The Curves to Build

### 3.1 Precision response surface (report-critical)

Fix everything except the target module. Keep the world model resident, keep embeddings BF16, and
quantize only the target linear layers. This isolates precision from deployment changes.

| Curve | Points | Trials per point | Purpose |
|---|---|---|---|
| Language | INT8, NF4, HQQ3 axis 0, HQQ2 axis 0 | 100 development; Q4/Q8 have 500 reserved anchors | Locate the decoder cliff |
| Vision | INT8, NF4, HQQ3 axis 0, HQQ2 axis 0 | 100 development | Test 2-bit tolerance |
| Action | INT8, NF4, HQQ3 axis 0, HQQ2 axis 1 | 100 development | Test 2-bit tolerance |
| Joint vision+action | Vision HQQ2 + action HQQ2, language BF16 | 100 development, 500 if it becomes a deployment claim | Test interaction |

Plot success against **effective bits**, not nominal bits:

\[
b_{\mathrm{eff}}(x) = \frac{8 \, S(x)}{N(x)},
\]

where \(S(x)\) is resident tensor storage for the module, including scales, zeros and metadata, and
\(N(x)\) is the logical parameter count. Nominal 3-bit HQQ can be worse than 4-bit NF4 because of
axis and group size; the report already records HQQ3 saving 2.3% storage over T2 but dropping
development success 87 to 81. Effective bits make that visible.

Primary comparisons, pre-registered:

- H1: language 2-bit loses more than 10 pp against B16.
- H2: vision 2-bit is non-inferior within 5 pp.
- H3: action 2-bit is non-inferior within 5 pp.
- H4: joint vision+action 2-bit is non-inferior within 5 pp; the interaction is not superadditive.

Use Wilson intervals per arm and a task-stratified paired bootstrap against B16. For the curve,
report measured points with intervals. Do not smooth the points in the report; a hierarchical model
is for choosing the next experiment, not for replacing measurements.

### 3.2 Format degradation

Decoder-only, same scope, 100-trial development for NF4, INT8, HQQ4, HQQ3, HQQ2, GPTQ and the
AWQ-style diagonal adaptation. Plot success against peak memory and effective bits. Keep Q4 and Q8
as 500-trial anchors.

This curve answers a different question from the bit curve: at the same nominal width, does the
format change the loss? It also prevents the report from attributing a bit-width effect to
"precision" when the real cause is group size, axis or calibration.

### 3.3 Execution ladder

Keep the existing ladder. It is the strongest part of the report: memory, stock latency, graph
latency, dense versus packed NF4, one versus four flow steps. Add one clarification: the 24.8 ms
B16 graph one-step number is a timing-only control; its success was not measured. Do not imply that
Dream-RSI discovered the engine. The engine is an independent execution intervention.

### 3.4 Search and joint recipes

Present the joint recipes as the outcome of the search, after the mechanism curves. A compact
search-history figure is better than a generic "Dream-RSI improved things" curve:

- x: resident MB.
- y: development success out of 100, with Wilson intervals.
- Points: Stage 2, R1, R3, R4, R5, and the C1/C2 baselines.
- Mark restricted-vocabulary points separately; they are not deployment candidates for the full
  vocabulary claim.

The figure should show the deterministic win: 89 to 125 MB smaller at overlapping development
intervals, not a significant success gain. That is the honest result.

### 3.5 Offline screening

The current sentence is correct. Make it concrete:

- Above \(10^{-3}\) action MSE, the screen is a strong rejector: ρ = -0.82.
- Between \(10^{-4}\) and \(10^{-3}\), it is weak: ρ = -0.25.
- Below \(10^{-4}\), it is uninformative: ρ = +0.19, p = 0.13.
- Below \(10^{-5}\), it is flat: ρ = 0.01.

So the metric is a cheap rejection test, not a ranking function. Near-lossless recipes need
closed-loop trials. This is exactly what the report says, but the numbers make it legible.

## 4. Formal Framework

### 4.1 Context-conditioned Pareto program

Let \(X\) be a finite set of admissible recipes. A context \(c \in C\) fixes the task suite, hardware,
flow steps, calibration data, trial budget and evaluation phase. Let

\[
F : X \times C \to \mathbb{R}^k
\]

map a recipe and context to \(k\) objectives, all oriented so larger is better: success, negative
peak memory, negative latency, negative training cost, and so on.

For fixed \(c\), define dominance:

\[
x \succ_c x' \quad \Longleftrightarrow \quad
F_i(x,c) \ge F_i(x',c) \ \forall i \quad \text{and} \quad
F_i(x,c) > F_i(x',c) \ \text{for some } i.
\]

The Pareto set is

\[
P(c) = \{ x \in X : \nexists x' \in X,\ x' \succ_c x \}.
\]

A requirement is \(r = (c, \tau)\), where \(\tau \subset \mathbb{R}^k\) is a target region. A program
is \(R = \{r_1, \ldots, r_m\}\). Its feasible set is

\[
X_R = \{ x \in X : F(x,c_j) \in \tau_j \ \forall j \}.
\]

The joint front is the Pareto set of the product objective over \(X_R\), not the union of the
individual fronts.

### 4.2 Why \(m(x)\) is ill-posed as stated

The proposal replaces the fixed context \(c\) with a function \(m(x)\), giving
\(F_m(x) = F(x,m(x))\). This is well-defined for any fixed \(m\), but it is not a meaningful
selection principle if \(m\) is free.

**Proposition (degeneracy).** If \(C\) is rich enough that for every \(x \in X\) there is a context
\(c_x\) with \(F(x,c_x) > F(x',c_x)\) for all \(x' \ne x\), and \(m\) may be chosen freely, then every
\(x \in X\) is Pareto-optimal under \(F_m\).

**Proof.** Set \(m(x) = c_x\). For any \(x' \ne x\), \(F(x,m(x)) > F(x',m(x))\). But dominance between
\(x\) and \(x'\) is evaluated at their own contexts, \(m(x)\) and \(m(x')\). To make \(x\) maximal it
is enough that no \(x'\) dominates \(x\) at \(m(x)\), which holds by construction. Therefore every
\(x\) is maximal. \(\square\)

The problem is not that \(F_m\) is undefined. The problem is that the order is not comparable across
candidates: each candidate can choose the context in which it wins. This is the same failure as
reporting each method on its own benchmark.

Admissible repairs:

1. **Evaluator-fixed context.** \(m(x) = c_0\) for all \(x\). This recovers standard Pareto. This is
   what the report should do: one context per front, with phase and hardware fixed.
2. **Monotone context.** Let \((C, \preceq_C)\) be a partial order, and let \(m\) be monotone with
   respect to a resource order on \(X\). Define

   \[
   x \succeq_m x' \quad \Longleftrightarrow \quad
   m(x) \succeq_C m(x') \ \text{and} \ F(x,m(x)) \ge F(x',m(x')).
   \]

   This is a partial order. It says: a candidate must be at least as good, and evaluated in a context
   at least as demanding. A candidate cannot dominate by choosing an easier context. Maximal
   elements exist for finite \(X\). If two contexts are incomparable, neither candidate dominates;
   the front is a partial order, not a total order. That is the honest representation.
3. **Learned context with uncertainty.** If \(m\) is a predictor of deployment context, it must be
   independent of the candidate's measured outcomes. Treat it as a covariate with uncertainty. The
   plug-in front is consistent if the objective estimator is uniformly consistent and the dominance
   relation is closed. This is a standard uniform-convergence argument; it does not justify letting
   the candidate choose \(m\).

### 4.3 Continuity, monotonicity, differentiability

**Continuity (Berge).** If \(X\) is compact and \(F\) is continuous, the Pareto correspondence
\(P : C \rightrightarrows X\) is upper hemicontinuous and compact-valued. Small changes in context do
not produce arbitrary jumps in the front. This is what makes a "front as a function of context" a
well-defined object.

**Monotonicity (Topkis).** If \(X\) and \(C\) are lattices, \(F\) has increasing differences in
\((x,c)\), and the feasible set is a sublattice, then maximal selections are monotone in \(c\).
Relaxing a memory budget cannot reduce the achievable success front; tightening it cannot improve
it. This justifies monotone claims such as "more memory cannot hurt success, all else fixed."

**Differentiability.** If \(F\) is \(C^1\) and the active constraints satisfy linear independence,
the Pareto set is locally a \(C^1\) manifold and the value function has shadow prices. This is the
sense in which the trade-off curve is differentiable. The actual LIBERO success is Bernoulli and is
not differentiable. Only a surrogate is. Do not claim end-to-end differentiability from rollouts.

### 4.4 The joint front is not the union of walls

The proposal speaks of three walls \(f(x,3), g(x,3), k(x,3)\). If those are three objectives, the
joint front is not their union.

**Counterexample.** Let \(X = \{x,y,z\}\) and two objectives, both larger-is-better:

\[
F(x) = (1,0), \quad F(y) = (2,-1), \quad F(z) = (-1,2).
\]

- On objective 1 alone, \(y\) wins; on objective 2 alone, \(z\) wins.
- \(x\) is not on either individual wall.
- But \(x\) is on the joint front: \(y\) is worse on objective 2, and \(z\) is worse on objective 1.

So the union of the individual walls misses joint Pareto points. The overseer must evaluate the joint
objective. The individual walls are diagnostic slices, not a complete solution. This is the main
correction to the "fgk" composition idea.

### 4.5 Phase-indexed soundness

Index every measurement by its phase:

\[
\Gamma \vdash (x, c, p, y, n), \quad p \in \{\text{screen}, \text{smoke}, \text{development}, \text{reserved}\}.
\]

Promotion rules may use smoke and development. Claim rules may use reserved, or a pre-registered
development claim. The reserved phase never appears in a selection premise.

Under these rules, any derived claim is either:

- a sample statement about measured trials, or
- a held-out statement about a policy selected without reserved data.

A development number cannot be typed as a held-out claim, because the typing rule does not allow it.
This is a syntactic guarantee. The statistical guarantee, that the reserved estimate is unbiased for
the selected policy, additionally requires that selection is independent of reserved data. The type
system enforces the independence; the estimator supplies the interval.

The Dream-RSI replay layer is a separate judgment:

\[
\Gamma \vdash V(\pi, T),
\]

where \(T\) is a frozen discovery tree. This judgment does **not** imply
\(\Gamma \vdash \text{success}(x)\). Replay value and LIBERO success are different types. Mixing them
is the main fidelity error to avoid.

### 4.6 Audit triad and judge fusion

The repo already has a 2-proposer, 3-judge, 1-fuser process in `docs/ORCHESTRATION.md`. Extend it
with the positive/neutral/negative pattern:

- **Positive (advocate):** strongest case for the move, using only measured evidence.
- **Neutral (outsider):** does the claim follow? Are the numbers traceable? Is the phase correct?
- **Negative (hostile):** what would make this false? Confounders, leakage, multiplicity, missing
  baseline, cost.
- **Judge:** returns one of `approve`, `fail`, `mix`, `prune`. `mix` accepts the move with named
  modifications. `prune` drops named claims or points. The judge records dissent.

Soundness:

- If each hard gate is sound for property \(P_i\), and the judge requires all hard gates to approve,
  the fused decision is sound for \(\bigwedge_i P_i\). Proof: conjunction introduction.
- Majority voting is not automatically sound. Condorcet's jury theorem requires conditional
  independence and accuracy above 1/2. Three prompts to the same model are correlated. Use different
  evidence slices, different roles, and a judge that resolves conflicts. Do not claim independence
  just because there are three reviews.

A proper audit does not "tend to give positive results." It gives calibrated results. If most ideas
are false, a good audit mostly rejects. Optimize for soundness (no false claims) over completeness
(approving every good idea).

### 4.7 Refined historical tree: evidence versus decision

Keep two trees, not one:

- **Evidence tree.** Immutable measured nodes: recipe, context, phase, trial count, successes,
  intervals, run path, code revision. Dream-RSI replay reads this tree.
- **Decision tree.** Proposals, scouts, audits, judge resolutions, pruned branches, corrections.
  The overseer and the meta-policy read this tree.

The audit appends `audit.json` and `judge.json` to a decision node. It never edits a measured node.
A correction is a new node that references the old one. This preserves the historical record and
matches the board's append-only rule.

The refined tree is useful, but do not let audit verdicts become replay scores. Replay remains
Eq. (1) over measured scores and gate classes. The audit may set a claim status
(`supported`, `overstated`, `false`, `mix`, `prune`) and may block promotion, but it must not become
a new objective. That would mix judgment with measurement and make the replay value uninterpretable.

Use the audit at promotion and claim nodes, not every smoke. It is a gate, not a per-trial tax.

## 5. Dream-RSI Implementation Plan

### 5.1 What already exists

- `rsi/tree.py`: discovery tree and deterministic replay question.
- `rsi/objective.py`: Eq. (1) hypervolume minus parallelism penalty.
- `rsi/policies/`: initial and optimal policies, plus baselines.
- `rsi/dream.py`: replay benchmark and monotonic selection.
- `rsi/policy_dev.py`: LLM policy-code rewriting with AST audit and subprocess verification.
- `rsi/online_cycle.py`: online tree expansion, smoke promotion, cycle loop.
- `rsi/sandbox.py`: `bwrap` isolation and headless `pi`.
- `rsi/gate.py`: sequential smoke gate with exact binomial operating characteristics.
- `rsi/action_space.py`, `rsi/digest.py`: bounded recipe space and history digest.

Measured readiness:

- `uv run --no-sync pytest -q`: 90 passed.
- `ruff check` and `ruff format --check`: clean.
- `pi` on `PATH` (or `~/.bun/bin/pi`); `bwrap` at `/usr/bin/bwrap`.
- RTX 3090 24 GB, 1.3 GB used; checkpoint `735d9f6` cached.
- Replay benchmark: 41 worlds, 11 policies, 5 beta values in about 2 seconds. Replay is not the
  bottleneck.

### 5.2 What is missing

1. **Agent-generated exploration policies.** The 11-policy pool should be a baseline set, not the
   search mechanism. The active exploration policy must be produced by the policy-development agent,
   validated, and selected on held-out replay value. The hand-written `OptimalPolicy` is the
   incumbent seed and the control, not the candidate generator.
2. **Policy genome and mutation.** "Thousands of candidate policies" needs a parameterized policy
   family or generated policy code. Add a bounded genome (budget schedule, branch choice, beta,
   gate retry, exploration temperature, batch shape) plus AST-audited code mutations. The agent
   writes the policy; the validator checks the prefix-only API; replay scores it.
3. **Meta-policy over requirements.** The current policy chooses which tree node to expand. It does
   not choose which requirement or context to instantiate next.
4. **Statistical surrogate.** No hierarchical model, no expected hypervolume improvement, no
   multiplicity control. The current selection is replay value, not a posterior over success.
5. **Audit fusion and calibration.** No automated positive/neutral/negative review with a judge
   resolution, and no calibration set for measuring reviewer bias.
6. **Scout layer.** No read-only codebase scouts that map evidence paths, entry points and risks
   before a proposal. The OpenAI kernel pipeline uses scouts to narrow 40M lines to 7%; the same
   move applies to `rsi/`, `evaluation/compression/` and `report/plots/`.
7. **Parallel custom rollouts.** Compression rollouts are serial, about 7 to 8 seconds per trial.
   `evaluation/parallel_rollout.py` exists but is not wired to arbitrary compression recipes.
8. **Held-out confirmation.** No Dream-RSI recipe has a reserved-500 confirmation.
9. **End-to-end loop exercise.** The current `vla-rsi loop` path is canonical, but the last live
   rounds used a round runner that has since been removed. The loop must be exercised with a fake
   evaluator and then one real cycle.

### 5.3 Meta-overseer design

Add a thin layer, not a new framework:

- `rsi/program.py`: requirement dataclasses, context-conditioned front, effective bits.
- `rsi/surrogate.py`: hierarchical Beta-Binomial for success, log-normal models for memory and
  latency, expected hypervolume improvement.
- `rsi/audit.py`: review triad prompts, judge schema, deterministic validation.
- `rsi/meta.py`: meta-policy over requirement programs, replay over historical trees.

The meta-policy state is the history \(H_t\), the current front estimate \(\hat{P}_t\), and the
current program \(R_t\). Actions are: instantiate a requirement, propose a recipe, run a measurement,
prune a requirement, promote a candidate. The reward is hypervolume improvement plus information
gain minus cost, with hard penalties for phase leakage and unsupported claims.

This is faithful to Dream-RSI because the base model and evaluator stay fixed. The overseer changes
the exploration policy, not the evidence rules. It is a meta-level version of the same move.

### 5.4 Replay scale

The current benchmark replays 41 worlds, 11 policies and 5 beta values in about 2 seconds. A
parameterized family of 1,000 policies over the same worlds would take a few minutes on CPU. That is
cheap enough to use replay as the inner loop. The expensive loop remains LIBERO rollouts: 10 trials
about 5 minutes, 100 trials about 12 minutes, 500 trials about 1 hour per arm on the RTX 3090.

Do not describe replay as "negligible" without measuring it. It is cheap, but it is not free, and it
is not evidence of LIBERO success.

### 5.5 Agent-generated policies, not a hand-written pool

The paper's move is to rewrite the exploration policy, not to hand-write every policy. Implement it
in three layers:

1. **Baseline pool (controls only):** `InitialPolicy`, `OptimalPolicy` incumbent, greedy serial,
   round-robin, epsilon-greedy, two-stage gate, random seeds. These establish the replay baseline
   and the monotonic selection incumbent. They are not the search.
2. **Policy genome:** a bounded parameterization of the exploration policy. Suggested genes:
   initial branch width, per-round budget schedule, exploit/explore mix, beta, gate retry, batch
   shape, and a termination rule. The genome is JSON, so it is easy to mutate, validate and replay.
3. **Generated policy code:** the policy-development agent reads replay traces and writes an
   `OptimalPolicy` implementation. The AST audit and subprocess replay gate already exist. The agent
   must beat the incumbent on held-out replay value, not in-sample.

Generate candidates with MAP-Elites or a simple evolutionary loop over the genome, then let the
policy-development agent write code for the best genomes. This gives thousands of cheap replay
candidates without hand-writing thousands of policies. Keep the code-generation step because the
paper's contribution is executable policy improvement, not just hyperparameter search.

### 5.6 Scout layer

Use read-only scouts before a proposal, following the OpenAI kernel pipeline's split between scouts
and analysts:

- A scout owns one bounded area: `rsi/`, `evaluation/compression/`, `report/plots/`, or `tests/`.
- It reads files, traces entry points and data flow, and writes a structured map: files, functions,
  evidence paths, invariants, risks, open questions, and a proposed audit target.
- It does not edit, run GPU jobs, or write outside its scratch directory.
- A checker opens every cited file and classifies each claim as verified, latent or false. A
  confirmer re-runs one small probe before the map enters the decision tree.

The article's lesson is to narrow the audit surface, not to over-scaffold. Use three to five scouts
per round, cap their output, and skip the layer when the target is already familiar. The scout map
is evidence, not a plan; the proposer still owns the proposal.

### 5.7 Parallel rollouts for custom recipes

Wire `evaluation/parallel_rollout.py` into `evaluation/compression/study.py` behind a `--workers`
flag. The path is already close: `parallel_rollout` takes a policy, a pipeline factory, a result
directory and a manifest, and it batches `predict_action_chunk` across env workers.

Required checks before using it for evidence:

- **Equivalence:** one recipe, serial versus N=2 and N=4, same seeds and states. Successes must match
  exactly; steps may differ by a few because batching changes the diffusion RNG stream.
- **Mode consistency:** compare parallel against parallel, or serial against serial. Do not compare a
  parallel arm to a serial control. For reserved claims, use the serial path to match the existing
  500-trial protocol.
- **GPU guard:** keep the existing `check_gpu_guard` and backoff. Never kill another process.
- **Manifest:** pass the task manifest so `initial_state_hash` and instructions are correct.
- **Failure isolation:** a worker error must fail the run, not silently drop trials.

Expected gain: about 1.3 to 1.5x throughput at N=4 on the RTX 3090. Use it for smoke and development
search; keep reserved confirmation serial.

### 5.8 Reserved confirmation for Dream-RSI

Yes, this is the missing step. Freeze one full-vocabulary recipe before running:

- Candidate: `r4-c1-t15-v35-g32` (86/100 development, 1199.6 MB) if no newer committed full-vocab
  recipe supersedes it. If the in-flight `r7-c1-v3-t20-v40-g32` is committed and frozen by its
  owner, it is the better candidate (86/100, 1186.9 MB).
- Phase: reserved 500, same 50 states per task, serial path.
- Control: B16 reserved 500 already exists.
- Gate: report the paired difference and Wilson interval. Claim non-inferiority only if the paired
  95% interval stays above -5 pp; do not re-select after seeing reserved outcomes.
- Record the freeze time, recipe hash, code revision and checkpoint revision before the run.
- If it fails, report the negative result. That is more valuable than another development point.

This is about one hour of GPU time and closes the largest gap between the report's Dream-RSI section
and its evidence.

### 5.9 Finish the `vla-rsi` loop

The tests already cover a two-cycle loop with a fake evaluator
(`test_online_two_cycle_loop_lazy_prefix_and_resume`). Finish the operational path:

1. Add a `--fake-evaluator` flag to `vla-rsi loop` for a CPU end-to-end run. It should exercise
   `plan_grid`, online expansion, smoke evaluation, promotion, tree export, policy development and
   monotonic selection without a GPU.
2. Run one real cycle with a small budget: `--probe-budget 4 --max-parallelism 2 --vocab-mode full`,
   development only. Verify the `pi` sandbox, recipe validation, smoke records, `tree.json`,
   `live_cycle_manifest.json` and the selected policy path.
3. Add resume/idempotence checks: re-running a cycle must reuse completed nodes and not duplicate
   measurements.
4. Add cost and time guards: stop after N probes, N GPU minutes, or two consecutive runtime failures.
5. Record the active policy path and its replay provenance in the manifest.

Do this before any large search round. A broken loop invalidates the search evidence, not just the
schedule.

### 5.10 Is the surrogate/audit "implicit" feasible without instruction bias?

Separate the two mechanisms:

- **The hierarchical Bayesian surrogate is not an LLM prompt.** It is a statistical model fit to
  measured outcomes. It has priors and partial pooling, so it is biased toward smoothness and toward
  the population mean, but that is a modeling choice, not a few-shot instruction. It cannot create
  evidence; it only ranks the next experiment. Validate calibration on held-out rounds before
  trusting it.
- **The audit triad is an LLM prompt and is biased.** Positive/neutral/negative instructions prime
  the model, the judge can anchor, and three calls to the same model are correlated. Treat the triad
  as a structured red-team, not as three independent samples.

Bias controls:

- Blind reviewers to the author's conclusion and to each other.
- Give each reviewer a different evidence slice: raw measurements, code, or the claim text.
- Put deterministic checks first: numbers traceable, phase correct, baseline present, interval
  computed, reserved untouched. The LLM handles only judgment calls.
- Use a fixed rubric and require evidence keys, not prose.
- Calibrate on 20 to 30 historical claims with known status (supported, overstated, false). Measure
  false approval and false rejection for single review, triad, and deterministic-plus-triad.
- Keep the audit out of the primary evidence path. It gates promotion and claims; it does not change
  measurements.
- Record disagreement and dissent. Do not average it away.

The empirical literature on LLM judges is mixed: judges help on some reasoning tasks, but they show
position, verbosity and self-preference biases, and multi-agent debate can amplify correlated errors.
The plan should measure its own audit, not assume it helps.

### 5.11 Invention ledger

The paper specifies interfaces and contracts, not every implementation detail. Label each component
before it enters an evidence claim.

| Component | Source | Status |
|---|---|---|
| `plan_grid(context) -> GridPlan` interface | Dream-RSI Appendix D.2 | Paper-faithful interface |
| `OptimalPolicy.plan_grid` constants and rules | Old `rsi/policies/optimal.py`: 6 branches, 4 depth, +2 on plateau, +1 depth on deep improvement, `rounds_cap` formula | **Removed.** The clean run reads a static plan from `studies/program-v1/protocol.json` |
| `gate_retry` / `gate_near_miss` | Old `rsi/policies/optimal.py` | **Removed** from the clean policy |
| Sequential C0/confirm gate | `rsi/gate.py` | **Invented addition**; statistical efficiency, not paper |
| Smoke score = size reduction if passed, else 0 | `rsi/online_cycle.py` | **Repo choice**; evaluator score must be pre-registered |
| `promote_top_k` by `(resident_bytes, -successes)` | `rsi/online_cycle.py` | **Repo choice**; paper does not specify promotion order |
| AST audit and subprocess replay verification | `rsi/policy_dev.py` | **Safety addition**; keep |
| Virtual branch × attempt slots | `rsi/online_cycle.py` | Implementation detail; not a paper claim |

`plan_grid` is not invented as an interface. The adaptive rule was. The clean run uses the static
plan in `studies/program-v1/protocol.json`; `OptimalPolicy.plan_grid` now requires an explicit
configuration and raises without one. Do not describe any adaptive rule as Dream-RSI.

### 5.12 Clean-room reset: deletion and full rerun

The old compression and Dream-RSI evidence was deleted from the working tree on 2026-11-15. It
remains in git history at tag `pre-cleanroom-20261115`. No old result is reused, pooled, or cited.
There is no tiered rerun.

The new namespace is `studies/program-v1/`. It contains a frozen `protocol.json`, a README, and empty
`recipes/`, `measurements/` and `dream-rsi/` directories. Every arm is rerun under that protocol:
B16, Q4, Q8, the precision surface, the format curve, the joint recipes, the execution ladder, and
the Dream-RSI clean loop.

Required artifacts per run:

- `run.json`: command, code revision, checkpoint revision, recipe hash, environment, GPU, protocol,
  timestamps, exit status.
- `episodes.csv`: one row per trial with task, trial, initial-state hash, seed, success, steps, wall
  time, status.
- `timings.csv`: median, p95, repetitions, warmups.
- `summary.json`: successes, intervals, paired differences, peak memory, storage inventory.
- `recipe.json` and transformation events.
- Agent logs, sandbox logs and `tree.json` for Dream-RSI cycles.
- Hashes for any packed archive; delete the archive only after exporting the hash and size.

The report is stale until the clean runs land. It may not quote a pre-cleanroom number.

## 6. Readiness Assessment

| Component | Ready? | Evidence | Gap |
|---|---|---|---|
| Replay engine | Yes | 90 tests; 41-world benchmark in 1.97 s; faithful map in `docs/DREAM_RSI.md` | Baseline pool is 11; no policy genome or generator |
| Online cycle | Code ready, not exercised end-to-end here | `pi`, `bwrap`, GPU, checkpoint present; fake-evaluator test exists | Last live rounds used removed runner; add CLI fake-evaluator and run one real cycle |
| Compression harness | Yes | 208 summary rows; 500-trial arms; screens and development gates | Serial rollouts; no custom-recipe parallel path |
| Statistical layer | Partial | Wilson intervals; paired bootstrap; sequential gate | No hierarchical model, EHVI, or multiplicity control |
| Audit layer | Partial | 2P→3J→1F in `docs/ORCHESTRATION.md` | No automated triad, judge schema, or calibration set |
| Scout layer | No | OpenAI kernel pipeline pattern; `docs/ORCHESTRATION.md` scout role | No read-only codebase map or checker |
| Report pipeline | Yes | `vla-report-plots` rebuilds figures and `numbers.json` | No precision/format figure code |
| Evidence discipline | Yes | Phase separation, reserved lock, compact records | New experiments need pre-registration |
| Hardware | Yes | RTX 3090 24 GB; 1.3 GB used | Shared card; coordinate before long runs |

## 7. Execution Plan

### Week 1 (Nov 16-22): precision curves

- Nov 16: Freeze the precision protocol and recipe list. Write
  `studies/program-v1/precision_surface_plan.json` with hypotheses, arms, phases, sample sizes,
  gates and analysis. Add the recipe JSONs. Do not run yet.
- Nov 17: Run 10-trial smoke screens for the new arms. Reject runtime failures and gross failures.
  Expected: 13 arms, about 1.5 hours.
- Nov 18-20: Run 100-trial development for language, vision, action and the joint 2-bit cell.
  Expected: 13 arms, about 3 hours of GPU time, spread over three days.
- Nov 21-22: Analyze with Wilson intervals and the paired bootstrap. Fit the hierarchical model.
  Build `report/plots/precision.py` and `report/figures/precision.pdf`. Draft the report section.

### Week 2 (Nov 23-29): formats, search, audit

- Nov 23-24: Run 100-trial development for the format curve. Expected: 5 to 7 arms, about 1.5 hours.
- Nov 25: Run one Dream-RSI cycle on the new precision axes, development only. Use the existing
  action space plus language bits, vision bits and action bits. Do not touch reserved trials.
- Nov 26-27: Build `rsi/program.py`, `rsi/surrogate.py` and `rsi/audit.py`. Backtest the surrogate on
  existing smoke/development data. Build the audit triad with fake reviewers and a deterministic
  judge.
- Nov 28-29: Replay benchmark with a parameterized policy family. Compare the meta-policy against
  the incumbent on held-out worlds. Write the formal note and update `docs/DREAM_RSI.md`.

### Week 3 (Nov 30-Dec 6): freeze, confirm, write

- Nov 30: Freeze the claims. Decide which arms, if any, get reserved-500 confirmation. Run at most
  two reserved arms, about 2 hours.
- Dec 1-2: Rebuild `report/plots`, `report/figures` and `report/data/numbers.json`. Check every
  number against the study records. Enforce the 5-page limit.
- Dec 3: Build the presentation. Use the precision surface and the execution ladder as the two main
  visuals.
- Dec 4: Presentation.
- Dec 5-8: Final report polish, references, reproducibility, page check.
- Dec 9: Report due.

### Stop rules

- A 100-trial point with a Wilson interval wider than ±10 pp cannot support a non-inferiority claim.
  Run 500 or drop the claim.
- A recipe that fails smoke at 6/10 or worse stops; do not promote.
- A format with screen MSE above \(10^{-3}\) and smoke below 8/10 stops.
- Do not use reserved trials for selection. Freeze first.
- If GPU time slips, drop the format figure to a table before dropping the precision curve.
- If replay value improves but development success does not, report the negative result. Do not
  claim LIBERO success.

## 8. Report Changes

Keep the 5-page budget. Proposed structure:

1. **Introduction and method.** Add one paragraph on the context-conditioned front and effective
   bits. Define the phase discipline.
2. **Figure 1.** Architecture and recipe funnel. Keep.
3. **Figure 2.** Precision response surface: module curves plus the joint vision+action cell. Replace
   the current 4-panel structural sensitivity figure. Move structural results to a compact table or
   the study README.
4. **Figure 3.** Execution ladder. Keep, simplify labels, label the 24.8 ms control as timing-only.
5. **Figure 4.** Search history: resident MB versus development success, with intervals. Optional
   small panel for the offline screen. Replace the current divergence figure if space is tight.
6. **Table 1.** Reserved results. Keep.
7. **Text.** Joint recipes after the mechanism curves; engine as the deployment step; backbone
   replacement as a boundary result.

Abstract rewrite, provisional:

> We measure post-training compression of VLA-JEPA on LIBERO-Spatial without retraining. A
> 100-trial precision surface shows the language decoder loses success at 2 bits, while vision and
> action remain non-inferior down to 2 bits; decoder-only NF4 cuts peak memory by 37% and stays
> within the 5-point margin. Two joint recipes reach 96.8-98.0% at 1.38-1.51 GB, with gains
> concentrated on one hard task. Stock execution is launch-bound; a CUDA-graph engine brings packed
> NF4 to 31.6 ms against 24.8 ms for the uncompressed policy at one flow step. A cheap action-change
> screen rejects broken recipes above \(10^{-3}\) but cannot rank near-lossless ones. Replay-guided
> search finds a full-vocabulary recipe 89 MB smaller at matched development success; it is not yet
> held-out confirmed. Replacing the backbone fails.

This is longer than the current abstract. Cut the execution sentence or the search sentence to fit.

## 9. Immediate Next Actions

1. Write the invention ledger into `docs/DREAM_RSI.md` and the new study README. Mark
   `plan_grid`'s 6/4/plateau rule, `gate_retry`, the sequential gate, the smoke score and
   `promote_top_k` as additions, not paper claims.
2. Freeze the clean-room protocol in `studies/program-v1/README.md` and a pre-registration JSON. No
   GPU run before this.
3. Old namespace deleted from the working tree. Recovery tag: `pre-cleanroom-20261115`. Do not
   import or cite it.
4. The 13 recipe JSONs are authored. Keep `world_remove` out of the module curves.
5. Run the 10-trial smoke screens. Record runtime failures separately from task failures.
6. Run the language curve first; it is the bottleneck claim and the most informative.
7. Build the precision figure from measured points, not from the model.
8. Run the replay benchmark with the current pool to establish the baseline before adding a policy
   genome. Keep the hand-written pool as controls only.
9. Add the policy genome and a MAP-Elites/evolutionary loop over it. Replay-score thousands of
   candidates on CPU; send only the best genomes to code generation.
10. Add `--fake-evaluator` to `vla-rsi loop`, then run one small real cycle. Verify resume, manifests
    and policy selection.
11. Wire `evaluation/parallel_rollout.py` into `vla-compress` behind `--workers`; run the serial
    versus parallel equivalence check before using it for development evidence.
12. Freeze one full-vocabulary recipe and run one reserved-500 confirmation under the clean protocol.
13. Write the audit triad schema, the judge output enum, and the 20-30 claim calibration set. Test
    one supported, one overstated and one false claim.
14. Send three to five read-only scouts over `rsi/`, `evaluation/compression/`, `report/plots/` and
    `tests/`; checker verifies every cited file before the map enters the decision tree.
15. Coordinate with the board before shared GPU work. Dream-RSI is Noah's area; post a CLM and keep
    the reserved-500 lock.

## 10. What This Plan Does Not Do

- It does not train the Qwen backbone.
- It does not touch reserved-500 trials for selection.
- It does not claim that replay value is LIBERO success.
- It does not prove that a free \(m(x)\) is meaningful; it proves the opposite and replaces it with
  an evaluator-fixed or monotone context.
- It does not claim the joint front is the union of individual walls; it proves the union is
  insufficient.
- It does not claim that an audit tends to approve; it claims that a sound audit can be trusted when
  it does approve.
