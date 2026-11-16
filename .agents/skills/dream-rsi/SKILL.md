---
name: dream-rsi
description: Paper specification (arXiv:2609.14858v2), fidelity boundaries, and engineering invariants for developing rsi/ in csi5341-vla-jepa. Use when modifying, designing, debugging, or describing Dream-RSI replay, policies, proposers, or tree expansion.
---

# Dream-RSI Development

Ground truth for `rsi/` architecture and paper fidelity: [docs/DREAM_RSI.md](../../../docs/DREAM_RSI.md).
Read it before editing `rsi/` or writing a claim about Dream-RSI. Running live GPU rounds: the
`auto-research` skill.

## Shape of a cycle

- **Online expansion ([rsi/expansion/](../../../rsi/expansion/), [rsi/online_question.py](../../../rsi/online_question.py)):**
  `probe_batch()` forks each chosen parent's frozen snapshot ([rsi/workspace.py](../../../rsi/workspace.py)),
  runs one discovery agent in `bwrap`, requires `proposal.md` and a top-level `compress`, freezes the
  workspace, and scores it with the fixed evaluator. The agent sees `/tmp/evidence` (`tree.json`, plus
  each earlier node's `proposal.md`, `score.json`, `error.txt`). Timeout, invalid output or an agent
  exit becomes a failed node; the cycle continues.
- **Duplicate digests:** `DigestRegistry` rejects a workspace whose code digest matches one already
  claimed in the cycle, root template included, without evaluation.
- **Concurrency:** up to `max_parallelism` expansions in threads. Every evaluation holds one of
  `dream_rsi.gpu_slots` flock slots ([rsi/gpu.py](../../../rsi/gpu.py)).
- **Policy rewriting ([rsi/policy_dev/](../../../rsi/policy_dev/)):** `develop_policies_llm()`
  makes `m-1` sequential rewrites of `OptimalPolicy` in `bwrap`, from `beta_sweep.json`,
  `policy_execution_traces.jsonl`, `live_cycle_manifests.json`, `history/` and frozen `worlds/`.
  Each candidate passes the AST audit, world-literal check and worker replay under two hash seeds;
  selection is strict improvement on in-sample V.
- **Replay $V \ne$ LIBERO success:** $V$ (Eq. 1) ranks exploration policies over frozen trees; it
  never establishes LIBERO task performance.

## Mandatory invariants

1. **Prefix-only policy contract (`rsi/policy_audit.py`, `rsi/api.py`):**
   - Policies may read only `question.observed()`, `legal_actions()`, `legal_roots()`,
     `opened_branches()`, `meta()`, `baseline_score`, `max_parallelism`, and call `probe_batch()`.
     `budget_spent`, `best_so_far`, `tree`, `roots`, `children` and other attributes are refused.
   - `audit_policy_source()` allows only `math` and the listed `from` imports, refuses any reference
     to builtins such as `open`, `getattr` and `eval` (not only calls), and refuses non-`self` private
     attributes. It also refuses `.format`, `reset`, module-level mutable state, dunder defs other
     than `__init__`/`__post_init__`, and classes other than `OptimalPolicy` and dataclasses.
   - Route thresholds through `_schedule(beta)`; keep `beta` fixed within `solve()`.
2. **Selection and out-of-sample checks (`rsi/policy_dev/develop.py`):**
   - Keep `m0_incumbent` at index `0`; `max` over `(mean_v, -step)` makes ties resolve to the
     incumbent.
   - A candidate replaces the incumbent only on strictly higher in-sample mean V.
   - Report leave-one-world-out whenever `len(history) >= 2`. Held-out synthetic worlds are reported,
     never used for selection.
3. **Phase and gate isolation (`rsi/runner.py`, `rsi/api.py`):**
   - Never mix smoke (10), development (100) and reserved (500) trials in one tree, world score or
     promotion count.
   - Never run `--phase final` from automated search or proposal loops. The node evaluator accepts
     only `check`, `smoke` and `development`.
   - Compare fail classes via `GATE_NEAR_MISS` / `GATE_FAILED` in `rsi/api.py`, not ad-hoc strings.
4. **Sandbox and secret safety (`rsi/sandbox/`, `rsi/agents/`):**
   - Keep the allowlist: `--unshare-all`, `--clearenv`, `--cap-drop ALL`. Repo source is mounted
     read-only only at `REPO_SOURCE_PATHS` under `/tmp/repo`. `CUDA_VISIBLE_DEVICES=""` only when GPU
     is off. Network only for built-in agents (`claude`, `pi`).
   - Never expose `.env`, `~/.claude/.credentials.json` or `~/.pi/agent/auth.json` outside the
     agent's read-only binds.
   - Replay runs in `rsi.replay_worker` inside `bwrap` with no network. Never default `--no-bwrap`
     for a run that counts.

## Verification

Pre-GPU check is the test suite; all of these must pass before any GPU run:

```sh
uv run --no-sync ruff check . && uv run --no-sync ruff format --check . && uv run --no-sync pytest -q -m "not slow"
```

Test files: `ls tests/test_rsi_*.py tests/test_solution_eval.py`.

Help-only checks (no GPU, no writes): `uv run --no-sync vla-rsi cycle --help` and
`uv run --no-sync vla-compress solution --help`.
