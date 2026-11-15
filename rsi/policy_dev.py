"""LLM replay-based policy-code rewriting and monotonic selection (Dream-RSI Appendix D.2)."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.api import LLMDesignedMethod, Observation
from rsi.dream import Candidate, leave_one_world_out, replay_terms, select_from_terms
from rsi.objective import (
    DEFAULT_BETA_GRID,
    DEFAULT_LAMBDA_PARALLEL,
    evaluate_policy_over_history,
)
from rsi.sandbox import REPO_ROOT, run_sandboxed_agent
from rsi.tree import DiscoveryTree, ReplayQuestion

INCUMBENT_TEMPLATE = Path(__file__).resolve().parent / "policies" / "optimal.py"

ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "dataclasses",
        "functools",
        "heapq",
        "itertools",
        "math",
        "random",
        "statistics",
        "typing",
    }
)
ALLOWED_FROM_PREFIXES = ("rsi.api",)
FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "hasattr",
        "input",
        "locals",
        "open",
        "setattr",
        "vars",
    }
)
FORBIDDEN_QUESTION_ATTRS = frozenset(
    {
        "best_so_far",
        "candidates",
        "children",
        "evaluator",
        "journal_path",
        "node_evals",
        "node_recipes",
        "roots",
        "tree",
    }
)


class PolicyAuditError(ValueError):
    """Raised when generated policy code violates the prefix-only or sandbox contract."""


def audit_policy_source(source: str) -> ast.Module:
    """Verify static AST safety, allowed imports, and prefix-only discipline."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PolicyAuditError(f"Syntax error in policy code: {exc}") from exc

    has_optimal_cls = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_MODULES:
                    raise PolicyAuditError(f"Forbidden import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root not in ALLOWED_MODULES and mod not in ALLOWED_FROM_PREFIXES:
                raise PolicyAuditError(f"Forbidden from-import {mod!r}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise PolicyAuditError(f"Forbidden builtin call {node.func.id!r}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id not in {"__future__"}:
                raise PolicyAuditError(f"Forbidden dunder reference {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_QUESTION_ATTRS:
                raise PolicyAuditError(
                    f"Prefix-only violation: accessing unallowed attribute {node.attr!r}"
                )
            if node.attr.startswith("_"):
                is_self = isinstance(node.value, ast.Name) and node.value.id == "self"
                is_super_init = (
                    node.attr == "__init__"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "super"
                )
                if not is_super_init and (not is_self or node.attr.startswith("__")):
                    raise PolicyAuditError(
                        f"Forbidden private/dunder attribute access {node.attr!r}"
                    )
        elif isinstance(node, ast.ClassDef) and node.name == "OptimalPolicy":
            has_optimal_cls = True
            methods = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "solve" not in methods:
                raise PolicyAuditError(
                    "OptimalPolicy must implement solve(self, question, budget=None)"
                )
            if "plan_grid" not in methods:
                raise PolicyAuditError("OptimalPolicy must override plan_grid(self, context)")

    if not has_optimal_cls:
        raise PolicyAuditError("Policy file must define class OptimalPolicy(LLMDesignedMethod)")
    return tree


def load_policy_class(policy_path: Path, unique_tag: str = "") -> type[LLMDesignedMethod]:
    """Audit and dynamically load `OptimalPolicy` from a standalone `policy.py`."""
    path = Path(policy_path).resolve()
    source = path.read_text(encoding="utf-8")
    audit_policy_source(source)

    mod_name = f"rsi_dynamic_policy_{path.stem}_{abs(hash((str(path), unique_tag)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise PolicyAuditError(f"Unable to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        raise PolicyAuditError(f"Failed importing policy {path}: {exc}") from exc

    cls = getattr(module, "OptimalPolicy", None)
    if cls is None or not isinstance(cls, type) or not issubclass(cls, LLMDesignedMethod):
        raise PolicyAuditError("OptimalPolicy must subclass rsi.api.LLMDesignedMethod")
    if getattr(cls, "NAME", None) != "OptimalPolicy":
        raise PolicyAuditError("OptimalPolicy.NAME must equal 'OptimalPolicy'")
    return cls


def verify_policy_in_subprocess(
    policy_path: Path,
    trees_json_paths: list[Path],
    *,
    timeout: float = 30.0,
) -> None:
    """Run static audit + isolated subprocess replay & plan_grid sanity check."""
    source = Path(policy_path).read_text(encoding="utf-8")
    audit_policy_source(source)

    probe_code = (
        "import json, resource, sys\n"
        "from pathlib import Path\n"
        "from rsi.api import GridPlan, GridPlanningContext, SimResult\n"
        "from rsi.policy_dev import load_policy_class\n"
        "from rsi.tree import DiscoveryTree, ReplayQuestion, TreeNode\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (20, 20))\n"
        "cls = load_policy_class(Path(sys.argv[1]), 'verify')\n"
        "for tp in sys.argv[2:]:\n"
        "    raw = json.loads(Path(tp).read_text())\n"
        "    nodes = [TreeNode(**{**n, 'tags': tuple(n.get('tags', ()))}) for n in raw['nodes']]\n"
        "    tree = DiscoveryTree(raw['world_id'], raw['baseline_score'], nodes)\n"
        "    for w in (1, 4):\n"
        "        for beta in (0.0, 0.5, 1.0):\n"
        "            for b_lim in (None, 4):\n"
        "                pol = cls({'beta': beta})\n"
        "                res = pol.solve(ReplayQuestion(tree, max_parallelism=w), budget=b_lim)\n"
        "                assert isinstance(res, SimResult)\n"
        "ctx = GridPlanningContext(iteration=1, max_parallelism=4, default_budget=12)\n"
        "plan = cls({'beta': 0.6}).plan_grid(ctx)\n"
        "assert isinstance(plan, GridPlan)\n"
        "assert plan.max_branches >= 1 and plan.max_attempts_per_branch >= 0\n"
        "assert plan.max_parallelism >= 1 and plan.rounds_cap >= 1\n"
    )
    cmd = [
        sys.executable,
        "-c",
        probe_code,
        str(Path(policy_path).resolve()),
        *[str(Path(p).resolve()) for p in trees_json_paths[:3]],
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()[-800:]
        raise PolicyAuditError(f"Policy subprocess verification failed:\n{err}")


def save_tree_json(tree: DiscoveryTree, path: Path) -> Path:
    """Persist a DiscoveryTree to JSON so replay worlds and subprocess verifiers share one schema."""
    payload = {
        "world_id": tree.world_id,
        "baseline_score": tree.baseline_score,
        "nodes": [
            {
                **asdict(tree.nodes[cid]),
                "tags": list(tree.nodes[cid].tags),
            }
            for cid in tree.order
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_tree_json(path: Path) -> DiscoveryTree:
    """Load a DiscoveryTree saved by `save_tree_json`."""
    from rsi.tree import TreeNode

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [TreeNode(**{**n, "tags": tuple(n.get("tags", ()))}) for n in raw["nodes"]]
    return DiscoveryTree(
        world_id=raw["world_id"],
        baseline_score=float(raw["baseline_score"]),
        nodes=nodes,
    )


class _TracingQuestion:
    """Thin proxy over ReplayQuestion that captures batch execution traces without leaking."""

    def __init__(self, inner: ReplayQuestion, world_id: str, beta: float) -> None:
        self._inner = inner
        self._world_id = world_id
        self._beta = beta
        self.events: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def reset(self) -> None:
        self._inner.reset()
        self.events.clear()

    def observed(self) -> dict[str, Observation]:
        return self._inner.observed()

    def legal_roots(self) -> list[str]:
        return self._inner.legal_roots()

    def opened_branches(self) -> list[int]:
        return self._inner.opened_branches()

    def legal_actions(self) -> list[str]:
        return self._inner.legal_actions()

    def meta(self, cell_id: str) -> Any:
        return self._inner.meta(cell_id)

    def probe_batch(self, cells: list[str], on_reveal: Any = None) -> list[Observation]:
        revealed_before = len(self._inner._revealed)
        out = self._inner.probe_batch(cells, on_reveal=on_reveal)
        self.events.append(
            {
                "world_id": self._world_id,
                "beta": self._beta,
                "round": self._inner.decision_rounds,
                "effective_sequential_rounds": self._inner.effective_sequential_rounds,
                "prefix_size": revealed_before,
                "batch": list(cells),
                "batch_size": len(cells),
                "max_parallelism": self._inner.max_parallelism,
                "outcomes": [
                    {
                        "cell_id": o.cell_id,
                        "branch": o.branch,
                        "attempt": o.attempt,
                        "parent_id": o.parent_id,
                        "fail_class": o.fail_class,
                        "score": round(o.score, 4) if o.score is not None else None,
                        "delta_vs_parent": (
                            round(o.delta_vs_parent, 4) if o.delta_vs_parent is not None else None
                        ),
                    }
                    for o in out
                ],
            }
        )
        return out


def export_policy_traces(
    policy_cls: type[LLMDesignedMethod],
    trees: list[DiscoveryTree],
    *,
    beta_sweep_path: Path,
    traces_path: Path,
    max_parallelism: int = 4,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    beta_grid: tuple[float, ...] = DEFAULT_BETA_GRID,
) -> dict[str, Any]:
    """Write `beta_sweep.json` and `policy_execution_traces.jsonl` for policy development."""
    sweep = evaluate_policy_over_history(
        policy_cls,
        trees,
        beta_grid=beta_grid,
        lambda_parallel=lambda_parallel,
        max_parallelism=max_parallelism,
    )
    beta_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    beta_sweep_path.write_text(json.dumps(sweep, indent=2) + "\n", encoding="utf-8")

    traces_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for tree in trees:
        for beta in beta_grid:
            pol = policy_cls({"beta": float(beta)})
            rq = ReplayQuestion(tree, max_parallelism=max_parallelism)
            tq = _TracingQuestion(rq, tree.world_id, float(beta))
            res = pol.solve(tq, budget=None)
            episode = {
                "world_id": tree.world_id,
                "beta": float(beta),
                "total_probes": res.total_probes,
                "decision_rounds": res.decision_rounds,
                "effective_sequential_rounds": res.effective_sequential_rounds,
                "best_score": res.best_score,
                "max_gain": tree.max_gain,
                "steps": tq.events,
            }
            lines.append(json.dumps(episode))
    traces_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return sweep


def _build_policy_dev_prompt(
    *,
    cycle_id: str,
    variant_index: int,
    focus_hint: str,
    max_parallelism: int,
    lambda_parallel: float,
) -> str:
    return (
        f"You are improving one **prefix-only exploration policy** for Dream-RSI cycle '{cycle_id}' "
        f"(candidate variant m={variant_index}).\n"
        f"Variant design emphasis: {focus_hint}\n\n"
        'Edit ONLY `/tmp/work/method.py` and keep `NAME = "OptimalPolicy"` and '
        "`class OptimalPolicy(LLMDesignedMethod)`. Do not solve the compression task and do not "
        "import filesystem, subprocess, network, or inspection modules.\n\n"
        "## Evidence to inspect first (read-only under `/tmp/evidence`)\n"
        "1. `/tmp/evidence/beta_sweep.json`: incumbent Eq. (1) reward, AUC, parallel penalty, and "
        "per-beta trade-offs.\n"
        "2. `/tmp/evidence/policy_execution_traces.jsonl`: episode-by-episode trace of batch "
        "selections and revealed outcomes at each decision round.\n"
        "3. `/tmp/evidence/live_cycle_manifests.json`: prior live cycle summaries (width/depth, "
        "opened branches, best score, baked-in beta).\n"
        "4. `/tmp/work/method.py`: current incumbent policy source.\n\n"
        "## Objective: quality, work, and parallelism (Eq. 1)\n"
        f"- Parallel worker capacity `W = {max_parallelism}`, penalty coefficient "
        f"`lambda = {lambda_parallel}`.\n"
        "- A batch of size `k` costs 1 decision round and `ceil(k / W)` effective sequential rounds.\n"
        "- Choose only promising probes, but batch independent promising probes up to "
        "`question.max_parallelism` instead of running serial singletons.\n"
        "- `gate_near_miss` (1 trial short of 10-trial smoke gate), `runtime_failure`, and "
        "`implementation_failure` are recoverable unless cumulative branch evidence indicates "
        "hard failure.\n\n"
        "## Mandatory prefix-only API & invariants\n"
        "- Allowed imports ONLY: `math`, `random`, `dataclasses`, `typing`, `collections`, "
        "`itertools`, `functools`, and `from rsi.api import ...`.\n"
        "- `solve(self, question, budget=None)` must read ONLY: `question.observed()`, "
        "`question.legal_actions()`, `question.legal_roots()`, `question.opened_branches()`, "
        "`question.meta(cell_id)`, `question.baseline_score`, `question.max_parallelism`.\n"
        "- NEVER access `question.best_so_far`, `question.budget_spent`, or private `_` attrs on "
        "any object other than `self`.\n"
        "- Keep `beta = float(self.config.get('beta', <chosen_default>))` fixed within `solve()`, "
        "routing all behavioral thresholds through `_schedule(beta) -> dict`. Note that offline "
        "replay sweeps `beta` over `(0.0, 0.25, 0.5, 0.75, 1.0)`, so merely changing the default "
        "`beta` without changing `_schedule(beta)` or `solve()` will tie with `m0_incumbent`.\n"
        "- Always implement deterministic `plan_grid(self, context: GridPlanningContext) -> GridPlan` "
        "using only `context` fields (`iteration`, `max_parallelism`, `recent_live_manifests`, "
        "`recent_beta_sweeps`).\n"
        "- `solve()` MUST terminate cleanly when `budget=None` or `not batch`.\n\n"
        "Update `/tmp/work/method.py` directly, optionally syntax/replay-check it with `python` "
        "(pure `rsi.api`, `rsi.tree`, `rsi.objective`, and `/tmp/evidence/worlds/*.json` are importable), "
        "and end with a concise summary of what changed and why."
    )


_VARIANT_HINTS = (
    "Improve failure/near-miss recovery priority vs. premature branch closure and balance root exploration.",
    "Improve parallel batch packing up to max_parallelism and refine _schedule(beta) sensitivity across beta in [0, 1].",
    "Improve trajectory scoring (parent-to-child gain momentum, diminishing returns stopping) and plan_grid width/depth allocation.",
)


def _rsi_ro_binds() -> dict[Path, str]:
    rsi_dir = Path(__file__).resolve().parent
    return {
        rsi_dir / "__init__.py": "/tmp/rsi_lib/rsi/__init__.py",
        rsi_dir / "api.py": "/tmp/rsi_lib/rsi/api.py",
        rsi_dir / "tree.py": "/tmp/rsi_lib/rsi/tree.py",
        rsi_dir / "objective.py": "/tmp/rsi_lib/rsi/objective.py",
    }


def develop_policies_llm(
    history: list[DiscoveryTree],
    *,
    cycle_id: str,
    incumbent_path: Path | None = None,
    output_dir: Path,
    live_manifests: list[dict[str, Any]] | None = None,
    m: int = 4,
    max_parallelism: int = 4,
    lambda_parallel: float = DEFAULT_LAMBDA_PARALLEL,
    agent_cmd: str | list[str] | None = None,
    model: str | None = None,
    thinking: str | None = None,
    timeout: float = 240.0,
    use_bwrap: bool = True,
) -> dict[str, Any]:
    """Run LLM policy rewriting (M-1 variants + incumbent) in bwrap sandboxes and select argmax V."""
    if not history:
        raise ValueError("Policy development requires at least 1 frozen DiscoveryTree in history")
    if m < 2:
        raise ValueError("m must be >= 2 (incumbent + at least 1 rewritten candidate)")

    inc_src = Path(incumbent_path) if incumbent_path is not None else INCUMBENT_TEMPLATE
    inc_cls = load_policy_class(inc_src, f"{cycle_id}_m0")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    evidence_dir = out_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Archive worlds inside evidence_dir/worlds so both verifier and sandboxed policy-dev inspect them
    world_json_paths = [
        save_tree_json(t, evidence_dir / "worlds" / f"{i:02d}_{t.world_id}.json")
        for i, t in enumerate(history)
    ]

    # Export incumbent sweep & execution traces into read-only evidence
    inc_sweep = export_policy_traces(
        inc_cls,
        history,
        beta_sweep_path=evidence_dir / "beta_sweep.json",
        traces_path=evidence_dir / "policy_execution_traces.jsonl",
        max_parallelism=max_parallelism,
        lambda_parallel=lambda_parallel,
    )
    (evidence_dir / "live_cycle_manifests.json").write_text(
        json.dumps(live_manifests or [], indent=2) + "\n", encoding="utf-8"
    )

    m0_dir = out_root / "m0_incumbent"
    m0_dir.mkdir(parents=True, exist_ok=True)
    m0_policy = m0_dir / "policy.py"
    shutil.copy2(inc_src, m0_policy)

    candidates: list[Candidate] = [
        Candidate("m0_incumbent", inc_cls, {}),
    ]
    candidate_paths: dict[str, Path] = {"m0_incumbent": m0_policy}
    variant_logs: list[dict[str, Any]] = [
        {
            "slot": "m0_incumbent",
            "status": "incumbent",
            "policy_path": str(m0_policy),
            "in_sample_reward": round(float(inc_sweep["pareto"]["reward"]), 6),
        }
    ]

    extra_binds = _rsi_ro_binds() if use_bwrap else None
    env_overrides = {"PYTHONPATH": "/tmp/rsi_lib"} if use_bwrap else None

    for idx in range(1, m):
        slot_name = f"m{idx}_llm"
        slot_dir = out_root / slot_name
        ws_dir = slot_dir / "workspace"
        logs_dir = slot_dir / "logs"
        ws_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        method_file = ws_dir / "method.py"
        shutil.copy2(inc_src, method_file)
        hint = _VARIANT_HINTS[(idx - 1) % len(_VARIANT_HINTS)]
        base_prompt = _build_policy_dev_prompt(
            cycle_id=cycle_id,
            variant_index=idx,
            focus_hint=hint,
            max_parallelism=max_parallelism,
            lambda_parallel=lambda_parallel,
        )

        valid_cls: type[LLMDesignedMethod] | None = None
        last_error: str | None = None
        attempts_log: list[dict[str, Any]] = []

        for attempt in (1, 2):
            prompt = (
                base_prompt
                if attempt == 1
                else (
                    f"{base_prompt}\n\n### Previous attempt validation failure\n{last_error}\n\n"
                    "Fix every violation in `/tmp/work/method.py` cleanly."
                )
            )
            res = run_sandboxed_agent(
                prompt_text=prompt,
                workspace_dir=ws_dir,
                evidence_dir=evidence_dir,
                logs_dir=logs_dir / f"attempt_{attempt}",
                agent_cmd=agent_cmd,
                model=model,
                tools="read,write,edit,bash",
                thinking=thinking,
                timeout=timeout,
                use_bwrap=use_bwrap,
                extra_ro_binds=extra_binds,
                env_overrides=env_overrides,
            )
            if res.returncode != 0:
                last_error = f"Agent exited {res.returncode}: {res.stderr[-300:]}"
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "valid": False,
                        "error": last_error,
                        "elapsed_s": res.elapsed_seconds,
                    }
                )
                continue
            try:
                verify_policy_in_subprocess(method_file, world_json_paths)
                valid_cls = load_policy_class(method_file, f"{cycle_id}_{slot_name}")
                attempts_log.append(
                    {"attempt": attempt, "valid": True, "elapsed_s": res.elapsed_seconds}
                )
                break
            except (PolicyAuditError, OSError, subprocess.TimeoutExpired) as exc:
                last_error = str(exc)
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "valid": False,
                        "error": last_error,
                        "elapsed_s": res.elapsed_seconds,
                    }
                )

        final_policy_path = slot_dir / "policy.py"
        if valid_cls is not None:
            shutil.copy2(method_file, final_policy_path)
            candidates.append(Candidate(slot_name, valid_cls, {}))
            candidate_paths[slot_name] = final_policy_path
            variant_logs.append(
                {
                    "slot": slot_name,
                    "status": "valid",
                    "policy_path": str(final_policy_path),
                    "attempts": attempts_log,
                }
            )
        else:
            variant_logs.append(
                {
                    "slot": slot_name,
                    "status": "rejected",
                    "error": last_error,
                    "attempts": attempts_log,
                }
            )

    world_ids = [t.world_id for t in history]
    terms = replay_terms(candidates, history, max_parallelism)
    in_sample_scores = {
        c.name: round(
            sum(terms[c.name][w][0] - lambda_parallel * terms[c.name][w][1] for w in world_ids)
            / float(len(world_ids)),
            6,
        )
        for c in candidates
    }
    selected_slot = select_from_terms(terms, world_ids, lambda_parallel)

    lowo_report: dict[str, Any] | None = None
    if len(history) >= 2:
        lowo_report = leave_one_world_out(terms, world_ids, lambda_parallel)

    # Monotonic selection guarantees selected_reward >= incumbent_reward on H_t
    winner_path = out_root / "selected_policy.py"
    shutil.copy2(candidate_paths[selected_slot], winner_path)

    # Read default beta baked into winning policy instance
    winner_cls = next(c.cls for c in candidates if c.name == selected_slot)
    winner_beta = float(getattr(winner_cls({}), "beta", 0.6))

    summary = {
        "protocol": "dream-rsi-policy-dev-v2",
        "cycle_id": cycle_id,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "num_worlds": len(history),
        "world_ids": world_ids,
        "max_parallelism": max_parallelism,
        "lambda_parallel": lambda_parallel,
        "selected_slot": selected_slot,
        "selected_policy_path": str(winner_path),
        "selected_default_beta": winner_beta,
        "incumbent_in_sample_V": in_sample_scores["m0_incumbent"],
        "selected_in_sample_V": in_sample_scores[selected_slot],
        "in_sample_V_by_slot": in_sample_scores,
        "lowo": (
            {
                "mean_selected_reward": round(float(lowo_report["mean_selected_reward"]), 6),
                "mean_incumbent_reward": round(float(lowo_report["mean_incumbent_reward"]), 6),
                "folds_selected_beats_incumbent": int(
                    lowo_report["folds_selected_beats_incumbent"]
                ),
                "folds_selected_below_incumbent": int(
                    lowo_report["folds_selected_below_incumbent"]
                ),
                "mean_held_out_reward_by_policy": {
                    k: round(float(v), 6)
                    for k, v in lowo_report["mean_held_out_reward_by_policy"].items()
                },
            }
            if lowo_report is not None
            else None
        ),
        "variants": variant_logs,
    }
    (out_root / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
