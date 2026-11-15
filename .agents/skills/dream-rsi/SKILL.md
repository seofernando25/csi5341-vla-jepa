---
name: dream-rsi
description: Paper specification (arXiv:2609.14858v2), fidelity boundaries, and engineering invariants for developing rsi/ in csi5341-vla-jepa. Use when modifying, designing, debugging, or describing Dream-RSI replay, policies, proposers, or tree expansion.
---

# Dream-RSI Development

Ground truth for `rsi/` architecture and paper fidelity lives in [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md). Read it before editing `rsi/` or writing any claim about Dream-RSI. Use `auto-research` for running live GPU rounds (`vla-rsi-propose` / `vla-rsi-round`); use this skill for modifying or reasoning about `rsi/` itself.

## Do Not Hallucinate Implementation Details

State only what the code actually does (see [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md) §2):

- **No online per-node LLM expansion yet:** `s2` and `r1`–`r6` use static pre-written chains (`rsi/qwen_search.py`, `rsi/live.py`). `r7` (`rsi/propose.py`) calls the LLM once *before* rollouts start and emits flat `attempt=0, parent_id="root"` roots (`propose.py:394-403`).
- **No LLM policy-code rewriting in `rsi/develop.py`:** `develop_policies()` replays `CANDIDATE_POOL` on CPU and writes 4 JSON config variants (`-tuned`, `-serial`, `-wide`, `-retry-flip`) of `OptimalPolicy` under `rsi/policies/revisions/`.
- **Single-GPU sequential probe execution:** `LiveQwenQuestion` sets `max_parallelism=4` for policy batching and Eq. (1) accounting, while `probe_batch()` executes `run_compression_subprocess()` sequentially (`qwen_search.py:423-428`).
- **`plan_grid()` is unwired live:** `OptimalPolicy.plan_grid()` is smoke-tested in `rsi/dream.py:543`, not called by `rsi/live_round.py`.
- **Replay $V \ne$ LIBERO success:** $V$ (Eq. 1) scores exploration policies over frozen trees using gate-qualified size reduction or historical scores; it never establishes LIBERO task performance.

## Mandatory Invariants

1. **Prefix-only policy contract (`rsi/api.py`, `rsi/tree.py`):**
   - Policies in `rsi/policies/` may read only `question.observed()`, `legal_actions()`, `legal_roots()`, `opened_branches()`, `meta()`, `baseline_score`, and `max_parallelism`.
   - Never branch on `question.best_so_far`, `question.budget_spent`, unrevealed nodes, or hardcoded cell IDs/targets.
   - Route every behavioral threshold in `OptimalPolicy` through `_schedule(beta)`; keep `beta` fixed within an episode.
2. **Monotonicity & out-of-sample checks (`rsi/dream.py`, `rsi/objective.py`):**
   - Keep `INCUMBENT` (`InitialPolicy`) at index `0` so argmax ties resolve conservatively to the incumbent.
   - Report leave-one-world-out (`leave_one_world_out`) alongside in-sample $V$, and run sensitivity/seed-stability grids; never pool suites or non-independent worlds without stating provenance.
3. **Phase & gate isolation (`rsi/gate.py`, `rsi/live.py`, `rsi/tree.py`):**
   - Never mix `smoke` (10), `development` (100), and `final` (500) trials in a `DiscoveryTree`, world score, or promotion count.
   - Never invoke `--phase final` from automated search or proposal loops.
   - Compare fail classes via constants in `rsi/api.py` (`GATE_NEAR_MISS`, `GATE_FAILED`), never ad-hoc strings.
4. **Action space & proposer safety (`rsi/action_space.py`, `rsi/propose.py`, `rsi/digest.py`):**
   - Keep `build_recipe_changes()`, `extract_knobs_from_changes()`, and `ACTION_SPACE` in lockstep; update canonical deduplication signatures when adding knobs.
   - Keep `require_half_full_vocab=True` (`C1` or `C2`) so proposals stay comparable to full-vocab baselines.
   - Never read, log, or commit `.env` secrets; journal `prompt_sha256` and `validation_log` on every proposal.

## Upgrading Toward Full Paper Fidelity

When implementing dynamic per-node tree expansion:

1. Preserve `r1`–`r6` builders and `r7` proposal journals as frozen replay provenance.
2. Expand at decision time: when `solve()` selects a node `v` (or `r`), pass `v`'s ancestor chain + observed sibling/world outcomes (`question.observed()`) to the proposer, validate the child against `ACTION_SPACE`, assign `parent_id=v`, `attempt=parent.attempt + 1`, and append to the live `DiscoveryTree`.
3. Keep CPU replay deterministic (`ReplayQuestion` must still replay the recorded online tree offline with zero LLM or GPU calls).

## Verification

After any edit to `rsi/`:

```sh
uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"
```

If replay worlds or `OptimalPolicy` logic changed, also verify CPU replay:

```sh
uv run --no-sync vla-rsi benchmark
```
