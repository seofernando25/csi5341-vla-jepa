# Dream-RSI Reference (`arXiv:2609.14858v2`) and `rsi/` Fidelity Map

Distilled specification of Tong Zheng et al., *"Dream-RSI: Recursive Self-Improvement through Evolving Worlds"* (`arXiv:2609.14858v2`, 2026-10-06) and exact comparison with `rsi/`.

## 1. Paper Architecture (Section 3, Appendix B.2, Appendix D)

The paper states **no numbered Algorithm**; the method is defined by Section 3, Eq. (1), Table 3, and Appendix D Listings 1–2.

Dream-RSI closes a recursive self-improvement loop at the **exploration-policy layer** ($\pi_t$), keeping the underlying base LLM, evaluator, and execution interfaces fixed:

1. **Discovery tree and batch interface:** A tree $\mathcal{T}$ is rooted at $r$ ($s_r = 0$). Each non-root node $v$ has one primary parent, resumes that parent's saved filesystem snapshot and history, runs the discovery agent once, stores its own snapshot and evaluation score $s_v$, and becomes a selectable leaf. Eligible actions are $A(\mathcal{T}) = \{r\} \cup \text{Leaves}(\mathcal{T})$; with $W \ge 1$ parallel workers, the exploration policy selects a batch $C \in A(\mathcal{T}; W) = \{C \subseteq A(\mathcal{T}) : |C| \le W\}$.
2. **Online rollout (Section 3; Appendix D.1 Listing 1):** At iteration $t$, policy $\pi_t$ drives a discovery agent for up to $K_1$ rounds. For each $v \in C_t^k$, a worker inspects every prior `proposal.md`, `eval/score.json`, and `error.txt` in full, writes a new `proposal.md` and `$eval_program`, evaluates it, and attaches the child node to $v$. When finished, $\mathcal{T}_t$ joins history $\mathcal{H}_t = \mathcal{H}_{t-1} \cup \{\mathcal{T}_t\}$.
3. **Offline replay ("dreaming", Section 3; Appendix D.2):** Each recorded $\mathcal{T}_i \in \mathcal{H}_t$ becomes a deterministic replay simulator (abstracted in Appendix D.2 as an irregular branch $\times$ attempt grid). Replaying policy version $\pi_t^m$ under budget-control knob $\rho \in \mathcal{R} = \{0, 0.25, 0.5, 0.75, 1\}$ reveals already-stored children $\text{Child}(v; \mathcal{T}_i, \mathcal{T}_{i,\rho}^{m,k})$ for up to $K_2$ rounds. No discovery agent or evaluator re-runs offline.
4. **Eq. (1) replay objective (Section 3):** Let $B_i = |\mathcal{T}_i| - 1$, $S_i = \max_{v \in \mathcal{T}_i \setminus \{r\}} s_v$, $n_{i,\rho}^m = |\mathcal{T}_{i,\rho}^{m,k_{i,\rho}^{m,\star}}| - 1$ calls, and $a_{i,\rho}^m = \text{clip}(s_{i,\rho}^m / S_i, 0, 1)$. Over the nondominated quality–cost frontier $(\bar{n}_{i,j}^m, \bar{a}_{i,j}^m)_{j=1}^{L_i^m}$ ordered by increasing $\bar{n}$ ($\bar{a}_{i,0}^m := 0$):
   $$V_i^m := \frac{1}{B_i}\sum_{j=1}^{L_i^m} (\bar{a}_{i,j}^m - \bar{a}_{i,j-1}^m)(B_i - \bar{n}_{i,j}^m) - \lambda \frac{1}{|\mathcal{R}|} \sum_{\substack{\rho \in \mathcal{R} \\ n_{i,\rho}^m > 0}} \frac{k_{i,\rho}^{m,\star}}{n_{i,\rho}^m}$$
   In Appendix D.2 Listing 2, `effective_sequential_rounds` increments by $\lceil |C| / W \rceil$ per batch, so serial batches score penalty $\approx 1$ and full batches approach $1/W$.
5. **Policy improvement and monotonic selection (Section 3; Appendix B.2; Appendix D.2):** Starting from $\pi_t^0 = \pi_t$, a **policy-development agent** inspects archived replay execution traces (`policy_execution_traces.jsonl`), `beta_sweep.json`, and `live_cycle_manifest.json` sidecars, then rewrites executable Python (`OptimalPolicy.solve` and `plan_grid`) into $\pi_t^{m+1}$ for $M-1$ steps ($M=4$ total in Table 3). Next deployment is $\pi_{t+1} = \pi_t^{m^\star}$ with $m^\star \in \arg\max_m \frac{1}{t}\sum_{i=1}^t V_i^m$, guaranteeing $V^{m^\star} \ge V^0$ on $\mathcal{H}_t$.

### Appendix D.2 Interface & `beta` Contract

- **Prefix-only invariant:** Decisions use only `question.observed()`, `legal_actions()`, `legal_roots()`, `opened_branches()`, `meta(cell_id)`, `baseline_score`, and `max_parallelism`. `best_so_far` and `budget_spent` are bookkeeping only.
- **Three roles of `beta` ($\rho$):** (1) fixed scalar inside `_schedule(beta)` during one episode; (2) swept across $\{0, 0.25, 0.5, 0.75, 1\}$ during offline replay; (3) updated once between live cycles from manifests and sweeps (default `0.6` when history is insufficient).
- **`plan_grid(context) -> GridPlan`:** Runs before creating a new live cycle to bound `(branch_count, refine_count)` capacity from prior manifests; it does not pre-generate the candidates inside those slots.

---

## 2. Fidelity Map: Paper vs. `csi5341-vla-jepa/rsi/`

| Component | Paper (`arXiv:2609.14858v2`) | Repo implementation | Status & exact gap |
|---|---|---|---|
| Decision interface | `see.policy` prefix-only API, `GridPlan` | [rsi/api.py](../rsi/api.py) | **Faithful** (`Observation`, `CellMeta`, `LLMDesignedMethod`; `GridPlan` fields `max_branches`, `max_attempts_per_branch`, `rounds_cap`) |
| Replay simulator | Deterministic prefix reveal over $\mathcal{T}_i$ | [rsi/tree.py](../rsi/tree.py), [rsi/policy_dev.py](../rsi/policy_dev.py) | **Faithful** (`DiscoveryTree`, `ReplayQuestion`, plus `tree.json` save/load for online cycles) |
| Replay objective | Eq. (1) hypervolume $-\ \lambda \cdot \text{penalty}$ | [rsi/objective.py](../rsi/objective.py) | **Faithful** ($\lambda=0.05$, $\beta \in \{0,0.25,0.5,0.75,1\}$, $\lceil \|C\|/W \rceil$) |
| Policy selection | $\arg\max_m V^m$ over $\mathcal{H}_t$ with $\pi_t^0$ | [rsi/dream.py](../rsi/dream.py), [rsi/policy_dev.py](../rsi/policy_dev.py) | **Faithful + stricter audits** (leave-one-world-out, $\lambda \times W$ grid, seed stability, AST + subprocess replay gate) |
| Exploration policy | Dynamic portfolio `OptimalPolicy` | [rsi/policies/optimal.py](../rsi/policies/optimal.py) | **Faithful** to Appendix D.2 Listing 2; adds `gate_retry` (`gate_near_miss` at 7/10) |
| Policy improvement | LLM inspects traces & rewrites `method.py` ($M=4$) | [rsi/policy_dev.py](../rsi/policy_dev.py) | **Faithful:** headless `pi` inside `bwrap` reads `beta_sweep.json`, `policy_execution_traces.jsonl`, `live_cycle_manifests.json`, and frozen worlds; rewrites `OptimalPolicy` with AST/prefix audit + 1 retry + monotonic selection. |
| Online tree expansion | Coding agent expands chosen $v \in C_t^k$ live | [rsi/online_cycle.py](../rsi/online_cycle.py), [rsi/sandbox.py](../rsi/sandbox.py) | **Faithful:** `OnlineDiscoveryQuestion.probe_batch()` expands chosen nodes at decision time via headless `pi` inside `bwrap`, inheriting `parent_context/`, enforcing full ordered recipe handlers (`validate_freeform_recipe`), and rejecting duplicates. |
| Worker concurrency | $W$ concurrent workers online | [rsi/online_cycle.py](../rsi/online_cycle.py) | **Split:** `bwrap` discovery agents run concurrently via `ThreadPoolExecutor`; compression evaluation remains sequential on 1 GPU while Eq. (1) accounts batch size against $W$. |
| Cross-cycle grid plan | `plan_grid` sizes next rollout capacity | [rsi/online_cycle.py](../rsi/online_cycle.py) | **Wired live:** `run_online_cycle()` calls `policy.plan_grid(...)` before building virtual slot capacity (`grid_plan.json`). |
| Archived Sep. harness | Discovery agent + policy-dev rewriting `policy.py` | Git commit `82df05d` (`studies/dream-rsi/attempts.jsonl`) | **Closer loop, wrong objective:** used SmolVLM held-out prediction loss (did not transfer to LIBERO success) |

---

## 3. Reporting & Provenance Boundaries

- **Provenance:** Records under `studies/dream-rsi/` keep their recorded semantics (static `r1`–`r6` grids, pre-round flat `attempt=0` roots for `r7`, and the deterministic config sweeps archived under `rsi/policies/revisions/`). Fresh full-loop runs live under `studies/dream-rsi-v2/` via `vla-rsi cycle`, `vla-rsi develop`, and `vla-rsi loop`.
- **Replay $V$ vs. LIBERO accuracy:** Replay objective $V$ scores exploration policies over frozen `DiscoveryTree`s using gate-qualified size reduction; actual task performance is measured only by LIBERO rollouts (`smoke`, `development`, and manual post-search `final`).
- **Sandbox boundary ([rsi/sandbox.py](../rsi/sandbox.py)):** Every headless `pi` discovery or policy-dev agent executes inside `bubblewrap` (`bwrap`) with the repository root masked (`--tmpfs`), `CUDA_VISIBLE_DEVICES=""`, read-only `/tmp/evidence/`, and writable scratch `/tmp/work/`.
