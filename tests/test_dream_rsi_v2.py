"""CPU tests for Dream-RSI v2: free-form recipe order, bwrap sandbox, LLM policy dev, online expansion."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from rsi.__main__ import main as rsi_main
from rsi.action_space import (
    canonical_recipe_signature,
    freeform_action_space_doc,
    validate_freeform_recipe,
)
from rsi.online_cycle import run_dream_rsi_loop, run_online_cycle
from rsi.policies import OptimalPolicy
from rsi.policy_dev import (
    PolicyAuditError,
    audit_policy_source,
    develop_policies_llm,
    export_policy_traces,
    verify_policy_in_subprocess,
)
from rsi.sandbox import SandboxTimeoutError, pi_argv, run_sandboxed_agent
from rsi.tree import DiscoveryTree, TreeNode

_HAS_BWRAP = shutil.which("bwrap") is not None


# ---------------------------------------------------------------------------
# 1. Free-form compression recipe schema & physical application order
# ---------------------------------------------------------------------------


def test_freeform_recipe_accepts_full_ordered_pipeline() -> None:
    raw = {
        "id": "iter_000-b0-a0",
        "family": "struct_wanda_hqq_graph",
        "engine": "graph",
        "engine_options": {"use_compile": True, "dense": False},
        "changes": [
            {"method": "world_remove"},
            {"method": "int8_embedding"},
            {"method": "channels", "scope": "vision_mlp", "fraction": 0.25},
            {"method": "heads", "scope": "text_attention", "fraction": 0.25},
            {"method": "layers", "scope": "text", "count": 2, "criterion": "cka"},
            {"method": "wanda", "scope": "action", "fraction": 0.3},
            {
                "method": "hqq",
                "scope": "action",
                "bits": 4,
                "axis": 1,
                "group_size": 64,
                "backend": "gemlite",
            },
            {"method": "bnb", "scope": "text", "bits": 4},
            {"method": "conditioning_tokens", "scope": "latent"},
            {"method": "solver_steps", "count": 4},
            {"method": "action_horizon", "count": 5},
            {"method": "replanning", "count": 3},
        ],
    }
    norm = validate_freeform_recipe(raw, expected_id="iter_000-b0-a0", vocab_mode="full")
    assert norm["engine"] == "graph"
    assert norm["engine_options"] == {"use_compile": True, "dense": False}
    assert len(norm["changes"]) == 12


@pytest.mark.parametrize(
    "bad_recipe,match",
    [
        # Quantization before channels on overlapping scope
        (
            {
                "id": "n1",
                "family": "bad_order",
                "changes": [
                    {"method": "bnb", "scope": "text", "bits": 4},
                    {"method": "channels", "scope": "text_mlp", "fraction": 0.2},
                ],
            },
            "must run BEFORE",
        ),
        # Wanda after channels on overlapping scope
        (
            {
                "id": "n2",
                "family": "bad_wanda",
                "changes": [
                    {"method": "channels", "scope": "text_mlp", "fraction": 0.2},
                    {"method": "wanda", "scope": "text", "fraction": 0.25},
                ],
            },
            "calibration",
        ),
        # AWQ after heads on overlapping scope
        (
            {
                "id": "n3",
                "family": "bad_awq",
                "changes": [
                    {"method": "heads", "scope": "action_attention", "fraction": 0.25},
                    {"method": "awq_diagonal", "scope": "action"},
                ],
            },
            "calibration",
        ),
        # Sparsity after quantization
        (
            {
                "id": "n4",
                "family": "bad_sparse",
                "changes": [
                    {"method": "bnb", "scope": "vision", "bits": 4},
                    {"method": "magnitude", "scope": "vision_mlp", "fraction": 0.4},
                ],
            },
            "BEFORE quantization",
        ),
        # sparse_kernel without 2:4
        (
            {
                "id": "n5",
                "family": "bad_sk",
                "changes": [
                    {"method": "sparsegpt", "scope": "text", "fraction": 0.5, "sparse_kernel": True}
                ],
            },
            "pattern='2:4'",
        ),
        # Quantizing a sparse_kernel linear
        (
            {
                "id": "n6",
                "family": "bad_sk_q",
                "changes": [
                    {
                        "method": "sparsegpt",
                        "scope": "text",
                        "pattern": "2:4",
                        "sparse_kernel": True,
                    },
                    {"method": "bnb", "scope": "text_mlp", "bits": 4},
                ],
            },
            "sparse_kernel=True",
        ),
        # Gemlite invalid axis / bits
        (
            {
                "id": "n7",
                "family": "bad_gem",
                "changes": [
                    {
                        "method": "hqq",
                        "scope": "text",
                        "bits": 3,
                        "axis": 0,
                        "group_size": 64,
                        "backend": "gemlite",
                    }
                ],
            },
            "gemlite backend",
        ),
        # visual_tokens incompatible with engine='graph'
        (
            {
                "id": "n8",
                "family": "bad_vt",
                "engine": "graph",
                "changes": [{"method": "visual_tokens", "fraction": 0.5}],
            },
            "engine='graph'",
        ),
        # compile_linears after all quantized
        (
            {
                "id": "n9",
                "family": "bad_comp",
                "changes": [
                    {"method": "bnb", "scope": "all", "bits": 4},
                    {"method": "compile_linears", "scope": "action"},
                ],
            },
            "unquantized",
        ),
        # replanning > action_horizon
        (
            {
                "id": "n10",
                "family": "bad_rep",
                "changes": [
                    {"method": "action_horizon", "count": 3},
                    {"method": "replanning", "count": 5},
                ],
            },
            "exceeds action_horizon",
        ),
    ],
)
def test_freeform_recipe_rejects_invalid_order_or_params(
    bad_recipe: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_freeform_recipe(bad_recipe)


def test_freeform_recipe_dedup_and_vocab_mode() -> None:
    r = {
        "id": "c1",
        "family": "hqq",
        "changes": [
            {"method": "world_remove"},
            {"method": "compact_embedding"},
            {"method": "bnb", "scope": "all", "bits": 4},
        ],
    }
    norm = validate_freeform_recipe(r, vocab_mode="restricted")
    sig = canonical_recipe_signature(norm)
    with pytest.raises(ValueError, match="duplicates"):
        validate_freeform_recipe(
            {**r, "id": "c2"},
            seen_signatures={sig: "c1"},
        )
    with pytest.raises(ValueError, match="vocab_mode='full'"):
        validate_freeform_recipe(r, vocab_mode="full")
    spec = json.loads(freeform_action_space_doc())
    assert "proper_order" in spec and "scopes" in spec


# ---------------------------------------------------------------------------
# 2. Bubblewrap sandbox isolation & headless `pi` argv
# ---------------------------------------------------------------------------


def test_pi_argv_is_extension_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rsi.sandbox.resolve_pi_executable", lambda: "/usr/local/bin/pi")
    argv = pi_argv(model="opencode-go/deepseek-v4.1-flash", tools="read,write")
    assert argv[0] == "/usr/local/bin/pi"
    for flag in ("--no-extensions", "--no-mcp", "--no-skills", "--no-session", "--print"):
        assert flag in argv
    assert "-p" in argv


@pytest.mark.skipif(not _HAS_BWRAP, reason="bubblewrap (bwrap) not installed")
def test_bwrap_sandbox_isolates_repo_and_evidence(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ev = tmp_path / "ev"
    logs = tmp_path / "logs"
    fake_repo = tmp_path / "fake_repo"
    fake_repo.mkdir()
    (fake_repo / ".env").write_text("SECRET=should_be_hidden\n", encoding="utf-8")

    probe = textwrap.dedent(
        f"""
        import json, os, sys
        from pathlib import Path

        hidden_env = Path({str(fake_repo / ".env")!r})
        assert not hidden_env.exists(), "repo_root .env leaked inside bwrap!"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        task_text = Path("/tmp/evidence/TASK.md").read_text()
        assert "hello evidence" in task_text

        ro_blocked = False
        try:
            Path("/tmp/evidence/tamper.txt").write_text("nope")
        except OSError:
            ro_blocked = True
        assert ro_blocked, "/tmp/evidence must be read-only!"

        Path("/tmp/work/out.json").write_text(json.dumps({{"ok": True, "len": len(task_text)}}))
        """
    ).strip()

    res = run_sandboxed_agent(
        prompt_text="hello evidence",
        workspace_dir=ws,
        evidence_dir=ev,
        logs_dir=logs,
        agent_cmd=[sys.executable, "-c", probe],
        timeout=15.0,
        use_bwrap=True,
        repo_root=fake_repo,
    )
    assert res.returncode == 0, res.stderr
    assert res.sandboxed is True
    out = json.loads((ws / "out.json").read_text(encoding="utf-8"))
    assert out == {"ok": True, "len": 14}


def test_sandbox_timeout_raises(tmp_path: Path) -> None:
    with pytest.raises(SandboxTimeoutError):
        run_sandboxed_agent(
            prompt_text="sleep",
            workspace_dir=tmp_path / "ws",
            evidence_dir=tmp_path / "ev",
            logs_dir=tmp_path / "logs",
            agent_cmd=[sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.3,
            use_bwrap=_HAS_BWRAP,
        )


# ---------------------------------------------------------------------------
# 3. Policy AST audit & LLM policy-code rewriting in bwrap
# ---------------------------------------------------------------------------


def _sample_trees() -> list[DiscoveryTree]:
    t0 = DiscoveryTree(
        world_id="w0",
        baseline_score=0.0,
        nodes=[
            TreeNode(
                "b0-a0", 0, 0, "root", 0, 0.70, True, True, "ok", None, "hqq", "ok", 0, (), 9, 10
            ),
            TreeNode(
                "b0-a1", 0, 1, "b0-a0", 1, 0.80, True, True, "ok", None, "hqq", "ok", 0, (), 9, 10
            ),
            TreeNode(
                "b1-a0",
                1,
                0,
                "root",
                2,
                0.0,
                True,
                False,
                "gate_near_miss",
                "7/10",
                "wanda",
                "miss",
                0,
                (),
                7,
                10,
            ),
            TreeNode(
                "b1-a1",
                1,
                1,
                "b1-a0",
                3,
                0.85,
                True,
                True,
                "ok",
                None,
                "wanda",
                "recovered",
                0,
                (),
                9,
                10,
            ),
            TreeNode(
                "b2-a0",
                2,
                0,
                "root",
                4,
                0.0,
                True,
                False,
                "gate_failed",
                "4/10",
                "layers",
                "fail",
                0,
                (),
                4,
                10,
            ),
        ],
    )
    t1 = DiscoveryTree(
        world_id="w1",
        baseline_score=0.0,
        nodes=[
            TreeNode(
                "x0-a0",
                0,
                0,
                "root",
                0,
                0.0,
                True,
                False,
                "gate_near_miss",
                "7/10",
                "awq",
                "miss",
                1,
                (),
                7,
                10,
            ),
            TreeNode(
                "x0-a1", 0, 1, "x0-a0", 1, 0.88, True, True, "ok", None, "awq", "ok", 1, (), 9, 10
            ),
            TreeNode(
                "x1-a0", 1, 0, "root", 2, 0.65, True, True, "ok", None, "bnb", "ok", 1, (), 8, 10
            ),
        ],
    )
    return [t0, t1]


@pytest.mark.parametrize(
    "bad_src",
    [
        "import os\nfrom rsi.api import LLMDesignedMethod\nclass OptimalPolicy(LLMDesignedMethod):\n NAME='OptimalPolicy'\n def solve(self,q,budget=None): pass\n def plan_grid(self,c): pass\n",
        "from rsi.api import LLMDesignedMethod\nclass OptimalPolicy(LLMDesignedMethod):\n NAME='OptimalPolicy'\n def solve(self,q,budget=None):\n  return q.best_so_far\n def plan_grid(self,c): pass\n",
        "from rsi.api import LLMDesignedMethod\nclass OptimalPolicy(LLMDesignedMethod):\n NAME='OptimalPolicy'\n def solve(self,q,budget=None):\n  return q._revealed\n def plan_grid(self,c): pass\n",
        "from rsi.api import LLMDesignedMethod\nclass OptimalPolicy(LLMDesignedMethod):\n NAME='OptimalPolicy'\n def solve(self,q,budget=None):\n  open('x')\n def plan_grid(self,c): pass\n",
    ],
)
def test_policy_audit_rejects_violations(bad_src: str) -> None:
    with pytest.raises(PolicyAuditError):
        audit_policy_source(bad_src)


def test_export_policy_traces_and_subprocess_verify(tmp_path: Path) -> None:
    trees = _sample_trees()
    sweep = export_policy_traces(
        OptimalPolicy,
        trees,
        beta_sweep_path=tmp_path / "beta_sweep.json",
        traces_path=tmp_path / "traces.jsonl",
    )
    assert "pareto" in sweep
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2 * 5  # 2 worlds x 5 betas
    wpath = tmp_path / "w0.json"
    from rsi.policy_dev import save_tree_json

    save_tree_json(trees[0], wpath)
    verify_policy_in_subprocess(Path("rsi/policies/optimal.py"), [wpath])


def test_develop_policies_llm_rewrites_and_selects_monotonically(tmp_path: Path) -> None:
    # Mock script inspects evidence/worlds & rsi.api inside bwrap and writes distinct policies per slot
    mock_agent = tmp_path / "mock_policy_agent.py"
    mock_agent.write_text(
        textwrap.dedent(
            """
            import json, os, textwrap
            from pathlib import Path
            from rsi.api import GridPlan, GridPlanningContext  # proves PYTHONPATH=/tmp/rsi_lib works!

            work = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
            ev = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
            task = (ev / "TASK.md").read_text()
            assert (ev / "beta_sweep.json").is_file()
            assert (ev / "policy_execution_traces.jsonl").is_file()
            assert list((ev / "worlds").glob("*.json")), "worlds missing in evidence"

            valid_rewritten = '''
            from rsi.api import (
                GATE_NEAR_MISS,
                GridPlan,
                GridPlanningContext,
                LLMDesignedMethod,
                SimResult,
                _budget_done,
                _record_curve,
                finalize_result,
            )

            class OptimalPolicy(LLMDesignedMethod):
                NAME = "OptimalPolicy"

                def __init__(self, config=None):
                    super().__init__(config)
                    self.beta = float(self.config.get("beta", 0.75))

                def _schedule(self, beta):
                    return {"batch_cap": max(2, round(2 + 2 * beta))}

                def solve(self, question, budget=None):
                    question.reset()
                    res = SimResult()
                    sched = self._schedule(self.beta)
                    while not _budget_done(question, budget):
                        obs = question.observed()
                        legal = question.legal_actions()
                        if not legal:
                            break
                        cap = min(question.max_parallelism, sched["batch_cap"])
                        if budget is not None:
                            cap = min(cap, max(0, budget - len(obs)))
                        # Prioritize children of ok or gate_near_miss branches, plus parallel roots
                        roots = set(question.legal_roots())
                        scored = []
                        for cid in legal:
                            meta = question.meta(cid)
                            p_obs = obs.get(meta.parent_id) if meta.parent_id != "root" else None
                            pri = 2.0 if (p_obs and p_obs.fail_class == GATE_NEAR_MISS) else (
                                1.5 if (p_obs and p_obs.valid) else (1.0 if cid in roots else 0.1)
                            )
                            scored.append((-pri, cid))
                        scored.sort()
                        batch = [cid for _, cid in scored[:cap]]
                        if not batch:
                            break
                        question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
                    return finalize_result(question, res)

                def plan_grid(self, context: GridPlanningContext) -> GridPlan:
                    return GridPlan(
                        max_branches=3,
                        max_attempts_per_branch=2,
                        max_parallelism=max(1, context.max_parallelism),
                        rounds_cap=4,
                        rationale="Rewritten greedy-parallel policy",
                    )
            '''.lstrip()

            if "variant m=1" in task and "Previous attempt validation failure" not in task:
                # Attempt 1 violates prefix-only audit -> triggers automatic repair attempt 2
                (work / "method.py").write_text(
                    "import os\\nfrom rsi.api import LLMDesignedMethod\\nclass OptimalPolicy(LLMDesignedMethod):\\n NAME='OptimalPolicy'\\n def solve(self,q,budget=None): pass\\n def plan_grid(self,c): pass\\n"
                )
            elif "variant m=3" in task:
                # Persistently invalid -> gets cleanly rejected without breaking selection
                (work / "method.py").write_text("import subprocess\\n")
            else:
                (work / "method.py").write_text(textwrap.dedent(valid_rewritten))
            """
        ).strip(),
        encoding="utf-8",
    )

    out_dir = tmp_path / "dev_out"
    res = develop_policies_llm(
        _sample_trees(),
        cycle_id="iter_000",
        output_dir=out_dir,
        live_manifests=[{"cycle_id": "iter_000", "best_score": 0.85}],
        m=4,
        max_parallelism=4,
        agent_cmd=[sys.executable, str(mock_agent)],
        timeout=30.0,
        use_bwrap=_HAS_BWRAP,
    )
    assert res["selected_in_sample_V"] >= res["incumbent_in_sample_V"]
    assert res["lowo"] is not None
    vmap = {v["slot"]: v for v in res["variants"]}
    assert vmap["m1_llm"]["status"] == "valid"
    assert len(vmap["m1_llm"]["attempts"]) == 2  # failed audit then repaired cleanly
    assert vmap["m2_llm"]["status"] == "valid"
    assert vmap["m3_llm"]["status"] == "rejected"
    assert (out_dir / "selected_policy.py").is_file()


# ---------------------------------------------------------------------------
# 4. Online per-node tree expansion + 2-cycle Dream-RSI v2 outer loop
# ---------------------------------------------------------------------------


def test_online_two_cycle_loop_lazy_prefix_and_resume(tmp_path: Path) -> None:
    discovery_agent = tmp_path / "mock_discovery.py"
    discovery_agent.write_text(
        textwrap.dedent(
            """
            import json, os
            from pathlib import Path

            work = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
            ev = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
            slot = json.loads((ev / "TARGET_SLOT.json").read_text())
            cid = slot["cell_id"]
            branch = slot["branch"]
            attempt = slot["attempt"]
            parent_id = slot["parent_id"]

            observed_names = sorted(p.name for p in (ev / "observed_nodes").iterdir())
            # Verify unrevealed nodes are never present in observed_nodes
            assert cid not in observed_names

            if attempt > 0:
                pctx = work / "parent_context"
                assert (pctx / "recipe.json").is_file(), "child must inherit parent_context"
                assert (pctx / "eval.json").is_file(), "child must inherit parent eval.json"

            # Record what was visible at expansion time to test online prefix property later
            (work / "expansion_audit.json").write_text(
                json.dumps({"observed_names": observed_names, "parent_id": parent_id})
            )

            # Force one invalid-order attempt on b0-a0 to exercise 1-retry validation repair
            trace_flag = work / ".bad_order_once"
            if cid.endswith("-b0-a0") and not trace_flag.exists():
                trace_flag.write_text("1")
                (work / "proposal.md").write_text("bad order attempt 1\\n")
                (work / "recipe.json").write_text(json.dumps({
                    "id": cid,
                    "family": "bad_order",
                    "changes": [
                        {"method": "bnb", "scope": "text", "bits": 4},
                        {"method": "channels", "scope": "text_mlp", "fraction": 0.2},
                    ],
                }))
                raise SystemExit(0)

            # Make distinct recipes per cycle & node so deduplication passes cleanly
            cnum = int(cid.split("_")[1].split("-")[0])
            frac = round(0.10 + 0.05 * ((branch + 2 * attempt + 3 * cnum) % 7), 2)
            scope = "text_mlp" if branch % 2 == 0 else "vision_mlp"
            steps = 3 + ((attempt + cnum) % 4)
            recipe = {
                "id": cid,
                "family": f"v2_{scope}_hqq",
                "engine": "graph",
                "engine_options": {"use_compile": True, "dense": False},
                "changes": [
                    {"method": "world_remove"},
                    {"method": "int8_embedding"},
                    {"method": "channels", "scope": scope, "fraction": frac},
                    {"method": "bnb", "scope": "text", "bits": 4},
                    {"method": "hqq", "scope": "action", "bits": 4, "axis": 1, "group_size": 64, "backend": "pytorch"},
                    {"method": "solver_steps", "count": steps},
                ],
            }
            (work / "proposal.md").write_text(f"# Proposal {cid}\\nparent={parent_id}, observed={observed_names}\\n")
            (work / "recipe.json").write_text(json.dumps(recipe, indent=2))
            """
        ).strip(),
        encoding="utf-8",
    )

    eval_calls: list[str] = []

    def fake_smoke(cid: str, recipe: dict[str, Any], _node_dir: Path) -> dict[str, Any]:
        eval_calls.append(cid)
        # Branch 0 gets 7/10 near-miss on root then 9/10 on refinement; others get small varying scores
        is_b0 = "-b0-" in cid
        attempt = int(cid.rsplit("-a", 1)[-1])
        if is_b0 and attempt == 0:
            return {
                "candidate": cid,
                "status": "completed",
                "successes": 7,
                "episodes": 10,
                "resident_bytes": 1_200_000_000,
                "failing_tasks": [2, 5, 8],
            }
        succ = 9 if (is_b0 or attempt > 0) else 8
        res_bytes = max(700_000_000, 1_500_000_000 - 150_000_000 * attempt)
        return {
            "candidate": cid,
            "status": "completed",
            "successes": succ,
            "episodes": 10,
            "resident_bytes": res_bytes,
        }

    def fake_promote(cid: str, recipe: dict[str, Any], _node_dir: Path) -> dict[str, Any]:
        return {
            "candidate": cid,
            "phase": "development",
            "status": "completed",
            "successes": 88,
            "episodes": 100,
            "resident_bytes": 1_050_000_000,
        }

    # Test single cycle + resume first, using a dedicated study folder
    study_dir = tmp_path / "study_v2"
    m0 = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study_dir,
        iteration=0,
        probe_budget=5,
        max_parallelism=2,
        promote_top_k=1,
        vocab_mode="full",
        agent_cmd=[sys.executable, str(discovery_agent)],
        agent_workers=2,
        use_bwrap=_HAS_BWRAP,
        evaluator=fake_smoke,
        promote_evaluator=fake_promote,
    )
    assert m0["probes_used"] == 5
    assert len(eval_calls) == 5
    assert len(m0["development_promotions"]) == 1
    assert (study_dir / "cycles" / "iter_000" / "tree.json").is_file()
    assert (study_dir / "cycles" / "iter_000" / "grid_plan.json").is_file()

    # Check lazy expansion: only probed slots exist, unprobed deep slots were never expanded
    nodes_dir = study_dir / "cycles" / "iter_000" / "nodes"
    expanded_slots = {p.name for p in nodes_dir.iterdir() if p.name != "root"}
    assert len(expanded_slots) == 5

    # Check prefix isolation in round 1 vs later rounds
    b0_a0_audit = json.loads(
        (nodes_dir / "iter_000-b0-a0" / "workspace" / "expansion_audit.json").read_text()
    )
    assert b0_a0_audit["observed_names"] == []  # first batch saw zero future nodes

    # Check resume safety: re-running iter_000 with same probe_budget performs zero extra GPU calls
    before_calls = len(eval_calls)
    m0_resumed = run_online_cycle(
        cycle_id="iter_000",
        study_dir=study_dir,
        iteration=0,
        probe_budget=5,
        max_parallelism=2,
        promote_top_k=1,
        vocab_mode="full",
        agent_cmd=[sys.executable, str(discovery_agent)],
        agent_workers=1,
        use_bwrap=_HAS_BWRAP,
        evaluator=fake_smoke,
        promote_evaluator=fake_promote,
    )
    assert m0_resumed["probes_used"] == 5
    assert len(eval_calls) == before_calls

    # Now verify full 2-cycle loop (online expansion -> LLM policy-dev -> cycle 1)
    loop_dir = tmp_path / "loop_v2"
    policy_script = tmp_path / "mock_pol_loop.py"
    policy_script.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path
            work = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
            # Copy incumbent template as-is for variant 1, and modify plan_grid slightly for variant 2
            src = (work / "method.py").read_text()
            if "plan_grid" in src:
                src = src.replace("base_branches = 6", "base_branches = 4")
            (work / "method.py").write_text(src)
            """
        ).strip(),
        encoding="utf-8",
    )

    # Note: run_dream_rsi_loop uses one agent_cmd for both phases if passed; test clean wiring
    # using a dispatcher that inspects TASK.md!
    dispatch_agent = tmp_path / "dispatch_agent.py"
    dispatch_agent.write_text(
        textwrap.dedent(
            f"""
            import os, runpy
            from pathlib import Path
            ev = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
            task = (ev / "TASK.md").read_text()
            if "policy_execution_traces.jsonl" in task:
                runpy.run_path({str(policy_script)!r}, run_name="__main__")
            else:
                runpy.run_path({str(discovery_agent)!r}, run_name="__main__")
            """
        ).strip(),
        encoding="utf-8",
    )

    loop_res = run_dream_rsi_loop(
        study_dir=loop_dir,
        cycles=2,
        probe_budget=4,
        max_parallelism=2,
        promote_top_k=1,
        vocab_mode="full",
        m_policy_variants=3,
        agent_cmd=[sys.executable, str(dispatch_agent)],
        agent_workers=2,
        use_bwrap=_HAS_BWRAP,
        evaluator=fake_smoke,
        promote_evaluator=fake_promote,
    )
    assert len(loop_res["cycles"]) == 2
    assert (loop_dir / "loop_summary.json").is_file()
    assert (loop_dir / "policies" / "iter_000" / "selection.json").is_file()
    # Second cycle saw prior_cycles/iter_000 in its evidence!
    c1_prior = (
        loop_dir
        / "cycles"
        / "iter_001"
        / "round_evidence"
        / "round_01"
        / "prior_cycles"
        / "iter_000"
    )
    assert (c1_prior / "live_cycle_manifest.json").is_file()


def test_parallel_batch_sibling_collision_retry_and_reject(tmp_path: Path) -> None:
    """Verify two parallel siblings returning the same recipe trigger re-expansion on the 2nd."""
    collide_agent = tmp_path / "collide_agent.py"
    collide_agent.write_text(
        textwrap.dedent(
            """
            import json, os
            from pathlib import Path

            work = Path(os.environ.get("RSI_WORKSPACE_DIR", "/tmp/work"))
            ev = Path(os.environ.get("RSI_EVIDENCE_DIR", "/tmp/evidence"))
            slot = json.loads((ev / "TARGET_SLOT.json").read_text())
            cid = slot["cell_id"]
            # b0-a0 emits fraction=0.2; b1-a0 emits 0.2 initially, then switches to 0.35 after duplicate rejection
            marker = work / ".retry_seen"
            if cid.endswith("-b1-a0") and marker.exists():
                frac = 0.35
            else:
                if cid.endswith("-b1-a0"):
                    marker.write_text("1")
                frac = 0.2
            recipe = {
                "id": cid,
                "family": "collide_test",
                "changes": [
                    {"method": "world_remove"},
                    {"method": "int8_embedding"},
                    {"method": "channels", "scope": "text_mlp", "fraction": frac},
                    {"method": "bnb", "scope": "text", "bits": 4},
                ],
            }
            (work / "proposal.md").write_text(f"# {cid}\\nfrac={frac}\\n")
            (work / "recipe.json").write_text(json.dumps(recipe))
            """
        ).strip(),
        encoding="utf-8",
    )

    m = run_online_cycle(
        cycle_id="iter_000",
        study_dir=tmp_path / "collide_study",
        probe_budget=2,
        max_parallelism=2,
        promote_top_k=0,
        vocab_mode="full",
        agent_cmd=[sys.executable, str(collide_agent)],
        agent_workers=2,
        use_bwrap=_HAS_BWRAP,
        evaluator=lambda cid, r, _: {
            "candidate": cid,
            "status": "completed",
            "successes": 9,
            "episodes": 10,
            "resident_bytes": 1_000_000_000,
        },
    )
    assert m["probes_used"] == 2
    r0 = json.loads(
        (tmp_path / "collide_study/cycles/iter_000/nodes/iter_000-b0-a0/recipe.json").read_text()
    )
    r1 = json.loads(
        (tmp_path / "collide_study/cycles/iter_000/nodes/iter_000-b1-a0/recipe.json").read_text()
    )
    assert r0["changes"][2]["fraction"] == 0.2
    assert r1["changes"][2]["fraction"] == 0.35


def test_rsi_cli_v2_subcommands_help() -> None:
    for sub in ("cycle", "develop", "loop"):
        with pytest.raises(SystemExit) as exc:
            rsi_main([sub, "--help"])
        assert exc.value.code == 0
