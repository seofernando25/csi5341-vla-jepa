---
name: dream-rsi
description: Paper specification (arXiv:2609.14858v2), fidelity boundaries, and engineering invariants for developing rsi/ in csi5341-vla-jepa. Use when modifying, designing, debugging, or describing Dream-RSI replay, policies, proposers, or tree expansion.
---

# Dream-RSI Development

Ground truth for `rsi/` architecture and paper fidelity lives in [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md). Read it before editing `rsi/` or writing any claim about Dream-RSI. Use `auto-research` for running live GPU rounds (`vla-rsi-propose` / `vla-rsi-round`); use this skill for modifying or reasoning about `rsi/` itself.

## Do Not Confuse Legacy v1 vs. Dream-RSI v2

Check [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md) §2 before writing claims:

- **Legacy v1 (`studies/dream-rsi/`, `r1`–`r7`):** `r1`–`r6` use static grids (`rsi/live.py`); `r7` (`rsi/propose.py`) uses pre-round flat root generation (`attempt=0, parent_id="root"`); `rsi/develop.py` runs deterministic 4-config CPU sweeps (`-tuned`, `-serial`, `-wide`, `-retry-flip`); `LiveQwenQuestion` does not invoke `plan_grid`.
- **Dream-RSI v2 (`studies/dream-rsi-v2/`, `vla-rsi cycle|develop|loop`):**
  - **True online per-node expansion ([rsi/online_cycle.py](../../../rsi/online_cycle.py)):** `OnlineDiscoveryQuestion.probe_batch()` expands chosen slots at decision time using headless `pi` inside `bwrap` ([rsi/sandbox.py](../../../rsi/sandbox.py)), inheriting `parent_context/` and seeing only the frozen revealed prefix (`observed_nodes/`).
  - **True LLM policy-code rewriting ([rsi/policy_dev.py](../../../rsi/policy_dev.py)):** `develop_policies_llm()` writes `M-1=3` `OptimalPolicy` variants inside `bwrap` from `beta_sweep.json`, `policy_execution_traces.jsonl`, `live_cycle_manifests.json`, and frozen `worlds/`, then runs AST/prefix + subprocess verification and monotonic selection.
  - **Free-form ordered recipes ([rsi/action_space.py](../../../rsi/action_space.py)):** `validate_freeform_recipe()` supports all 21 `evaluation/compression/recipes.py` handlers (excluding missing `torchao`) + `engine="graph"` (`use_compile`, `dense`) and enforces physical stage ordering (`world_remove`/embed -> structural -> sparsity -> quantization -> execution/compile).
- **Single-GPU evaluation:** Sandboxed discovery expands batches concurrently, while compression GPU smoke/dev subprocesses execute sequentially on 1 GPU.
- **Replay $V \ne$ LIBERO success:** $V$ (Eq. 1) scores exploration policies over frozen trees using gate-qualified size reduction; it never establishes LIBERO task performance.

## Mandatory Invariants

1. **Prefix-only policy contract (`rsi/api.py`, `rsi/tree.py`, `rsi/policy_dev.py`):**
   - Policies may read only `question.observed()`, `legal_actions()`, `legal_roots()`, `opened_branches()`, `meta()`, `baseline_score`, `max_parallelism`, and `budget_spent`.
   - `audit_policy_source()` enforces allowed stdlib + `rsi.api` imports and forbids `open`, OS/subprocess calls, `question.best_so_far`, `question.tree`, and non-`self` `_` attributes.
   - Route behavioral thresholds through `_schedule(beta)`; keep `beta` fixed within `solve()`.
2. **Monotonicity & out-of-sample checks (`rsi/dream.py`, `rsi/policy_dev.py`):**
   - Keep `m0_incumbent` (`INCUMBENT`) at index `0` so argmax ties resolve conservatively to the incumbent.
   - Report leave-one-world-out (`leave_one_world_out`) whenever `len(history) >= 2`.
3. **Phase & gate isolation (`rsi/gate.py`, `rsi/live.py`, `rsi/online_cycle.py`):**
   - Never mix `smoke` (10), `development` (100), and `final` (500) trials in a `DiscoveryTree`, world score, or promotion count.
   - Never invoke `--phase final` from automated search or proposal loops.
   - Compare fail classes via constants in `rsi/api.py` (`GATE_NEAR_MISS`, `GATE_FAILED`), never ad-hoc strings.
4. **Sandbox & secret safety (`rsi/sandbox.py`, `rsi/action_space.py`):**
   - Keep `repo_root` masked via `--tmpfs` inside `bwrap` and `CUDA_VISIBLE_DEVICES=""` set for all LLM workers; never expose `.env` or `~/.pi/agent/auth.json` outside `pi` auth ro-bind.
   - Keep `validate_freeform_recipe()` and `validate_change_order()` aligned with `evaluation/compression/recipes.py` and `evaluation/fastpath.py`.

## Verification

After any edit to `rsi/`:

```sh
uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"
```

If replay worlds or `OptimalPolicy` logic changed, also verify CPU replay:

```sh
uv run --no-sync vla-rsi benchmark
```
