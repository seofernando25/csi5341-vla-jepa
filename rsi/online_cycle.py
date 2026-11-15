"""Dream-RSI v2 online per-node discovery tree expansion inside `bwrap` + `pi`."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rsi.action_space import (
    canonical_recipe_signature,
    freeform_action_space_doc,
    recipe_vocab_type,
    validate_freeform_recipe,
)
from rsi.api import (
    CellMeta,
    GridPlan,
    GridPlanningContext,
    Observation,
    gate_fail_class,
)
from rsi.digest import _read_smoke_failing_tasks, _reference_points
from rsi.live import DEV_GATE, SMOKE_GATE, gpu_name, gpu_tight, wilson_interval
from rsi.policy_dev import (
    INCUMBENT_TEMPLATE,
    develop_policies_llm,
    load_policy_class,
    load_tree_json,
    save_tree_json,
)
from rsi.qwen_search import B16_RESIDENT_BYTES, run_compression_subprocess
from rsi.sandbox import run_sandboxed_agent
from rsi.tree import DiscoveryTree, TreeNode

DREAM_V2_DIR = Path("studies/dream-rsi-v2")
MEASUREMENTS_DIR = Path("studies/compression/measurements")
NUMBERS_PATH = Path("report/data/numbers.json")


def _extract_json_object(raw_text: str) -> Any:
    text = raw_text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        idx = text.find("{", pos)
        if idx == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = idx + 1
    raise ValueError("No valid JSON object found")


def _build_discovery_prompt(
    *,
    cell_id: str,
    branch: int,
    attempt: int,
    parent_id: str,
    vocab_mode: str,
    direction_guidance: str = "",
) -> str:
    parent_note = (
        f"You are creating a NEW root branch (slot `{cell_id}`, branch={branch}, attempt=0)."
        if parent_id == "root"
        else (
            f"You are REFINING/REPAIRING parent node `{parent_id}` "
            f"(slot `{cell_id}`, branch={branch}, attempt={attempt}). "
            "Read `/tmp/work/parent_context/` (inherited parent `proposal.md`, `recipe.json`, "
            "`eval.json`, and `error.txt` if present) before designing this child."
        )
    )
    vocab_rule = (
        "- Vocabulary mode is `full`: do NOT use `compact_embedding`; use `int8_embedding`."
        if vocab_mode == "full"
        else (
            "- Vocabulary mode is `restricted`: MUST use `compact_embedding`."
            if vocab_mode == "restricted"
            else "- Vocabulary mode is `any`: you may use `int8_embedding` (full vocab) or `compact_embedding` (restricted LIBERO-Spatial vocab)."
        )
    )
    guidance = f"\nDirection guidance:\n{direction_guidance}\n" if direction_guidance else ""

    return (
        f"You are the Dream-RSI online discovery agent for VLA-JEPA Qwen3-VL LIBERO-Spatial "
        f"compression.\n{parent_note}\n{guidance}\n"
        "## 1. Read the complete frozen evidence first (under `/tmp/evidence`)\n"
        "Before proposing anything, inspect ALL files under `/tmp/evidence/` in full:\n"
        "- `/tmp/evidence/baseline.json`: B16/C1/C2 reference baselines & gate definitions.\n"
        "- `/tmp/evidence/action_space.json`: supported compression handlers, scopes, "
        "`engine='graph'` options, and mandatory stage ordering rules.\n"
        "- `/tmp/evidence/observed_nodes/`: EVERY revealed node so far (`proposal.md`, "
        "`recipe.json`, `eval.json`, `error.txt`). Trust measured outcomes over claims.\n"
        "- `/tmp/evidence/prior_cycles/`: summaries of completed earlier Dream-RSI v2 cycles.\n"
        "- `/tmp/evidence/in_batch_slots.json`: sibling slots running in this parallel batch "
        "(choose a distinct mechanism from siblings when opening roots).\n\n"
        "## 2. Diagnose successes, failures, and avoid local optima\n"
        "- If refining a near-miss (`gate_near_miss`, 7/10 smoke) or failure (`runtime_failure` / "
        "`implementation_failure`), diagnose WHY from `eval.json`, failing task IDs, and `error.txt`, "
        "then propose a targeted fix or parameter adjustment.\n"
        "- If branches cluster in one mechanism with flattening returns, pick a genuinely different "
        "compression axis (e.g., pruning, HQQ/AWQ/GPTQ/SparseGPT, attention matching, token "
        "compaction, layer skipping, solver steps, `engine='graph'` with `dense=false`).\n"
        "- Respect physical stage order strictly: `world_remove` + embedding -> `channels`/`heads`/"
        "`layers` -> sparsity (`magnitude`/`wanda`/`sparsegpt`) -> quantization (`bnb`/`hqq`/"
        "`awq_diagonal`/`gptq`/`attention_matching`) -> execution (`visual_tokens`/`conditioning_tokens`/"
        "`solver_steps`/`action_horizon`/`replanning`/`compile_linears`).\n"
        f"{vocab_rule}\n\n"
        "## 3. Deliverable (write ONLY inside `/tmp/work`)\n"
        "Write exactly these two files in `/tmp/work/`:\n"
        "1. `/tmp/work/proposal.md`: concise factual motivation, evidence cited, expected memory/"
        "accuracy/latency effect, and why it is distinct from already-probed recipes.\n"
        f'2. `/tmp/work/recipe.json`: valid JSON object with exact `"id": "{cell_id}"`, '
        '`"family"`, optional `"engine"`/`"engine_options"`, and ordered `"changes"`.\n'
        "Do NOT execute GPU scripts or touch paths outside `/tmp/work`."
    )


def _format_eval_summary(
    *,
    cid: str,
    branch: int,
    attempt: int,
    parent_id: str,
    recipe: dict[str, Any] | None,
    raw_res: dict[str, Any],
    smoke_gate: int,
) -> tuple[dict[str, Any], str | None]:
    status = raw_res.get("status", "unknown")
    resident = raw_res.get("resident_bytes")
    resident_f = float(resident) if resident is not None else float(B16_RESIDENT_BYTES)
    vocab = recipe_vocab_type(recipe.get("changes", [])) if recipe else "unknown"

    if status != "completed":
        err = str(raw_res.get("error") or status)
        eval_doc = {
            "candidate": cid,
            "branch": branch,
            "attempt": attempt,
            "parent_id": parent_id,
            "vocab": vocab,
            "status": status,
            "fail_class": "runtime_failure" if status == "runtime_failure" else status,
            "passed_smoke_gate": False,
            "smoke_gate": smoke_gate,
            "score": 0.0,
            "error": err,
            "recorded_utc": datetime.now(UTC).isoformat(),
        }
        return eval_doc, err

    succ = int(raw_res["successes"])
    episodes = int(raw_res["episodes"])
    fail_class = gate_fail_class(succ, smoke_gate)
    passed = fail_class == "ok"
    reduction = max(0.0, 1.0 - resident_f / B16_RESIDENT_BYTES)
    score = round(reduction if passed else 0.0, 6)
    err = None if passed else f"smoke={succ}/{episodes}<{smoke_gate}"
    eval_doc = {
        "candidate": cid,
        "branch": branch,
        "attempt": attempt,
        "parent_id": parent_id,
        "vocab": vocab,
        "status": "completed",
        "successes": succ,
        "episodes": episodes,
        "wilson_95": wilson_interval(succ, episodes),
        "resident_bytes": round(resident_f),
        "resident_mb": round(resident_f / 1e6, 1),
        "size_reduction": round(reduction, 6),
        "fail_class": fail_class,
        "passed_smoke_gate": passed,
        "smoke_gate": smoke_gate,
        "score": score,
        "wall_seconds": raw_res.get("wall_seconds"),
        "failing_tasks": raw_res.get("failing_tasks"),
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    return eval_doc, err


class OnlineDiscoveryQuestion:
    """Dream-RSI question where each `probe_batch` dynamically expands chosen nodes via `pi` in `bwrap`."""

    def __init__(
        self,
        *,
        cycle_id: str,
        grid_plan: GridPlan,
        cycle_dir: Path,
        evaluator: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
        prior_cycle_dirs: list[Path] | None = None,
        smoke_gate: int = SMOKE_GATE,
        vocab_mode: str = "any",
        agent_cmd: str | list[str] | None = None,
        model: str | None = None,
        thinking: str | None = None,
        agent_timeout: float = 180.0,
        agent_workers: int = 2,
        use_bwrap: bool = True,
        direction_guidance: str = "",
        seen_signatures: dict[tuple[Any, ...], str] | None = None,
        numbers_path: Path = NUMBERS_PATH,
    ) -> None:
        self.cycle_id = cycle_id
        self.grid_plan = grid_plan
        self.cycle_dir = Path(cycle_dir)
        self.nodes_dir = self.cycle_dir / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.recipes_dir = self.cycle_dir / "recipes"
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.prior_cycle_dirs = list(prior_cycle_dirs or [])

        self.evaluator = evaluator
        self.smoke_gate = smoke_gate
        self.vocab_mode = vocab_mode
        self.agent_cmd = agent_cmd
        self.model = model
        self.thinking = thinking
        self.agent_timeout = agent_timeout
        self.agent_workers = max(1, agent_workers)
        self.use_bwrap = use_bwrap
        self.direction_guidance = direction_guidance
        self.numbers_path = Path(numbers_path)

        self.baseline_score = 0.0
        self.max_parallelism = grid_plan.max_parallelism

        # Build virtual branch x attempt slot skeleton from GridPlan
        self._slots: dict[str, CellMeta] = {}
        self.roots: list[str] = []
        self.children: dict[str, list[str]] = {}
        seq = 0
        for b in range(grid_plan.max_branches):
            parent = "root"
            for a in range(grid_plan.max_attempts_per_branch + 1):
                cid = f"{cycle_id}-b{b}-a{a}"
                cm = CellMeta(cell_id=cid, branch=b, attempt=a, parent_id=parent, seq=seq)
                self._slots[cid] = cm
                if parent == "root":
                    self.roots.append(cid)
                else:
                    self.children.setdefault(parent, []).append(cid)
                parent = cid
                seq += 1

        self.seen_signatures: dict[tuple[Any, ...], str] = dict(seen_signatures or {})
        self.node_recipes: dict[str, dict[str, Any]] = {}
        self.node_evals: dict[str, dict[str, Any]] = {}

        self._revealed: dict[str, Observation] = {}
        self._revealed_order: list[str] = []
        self.best_so_far = 0.0
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

        self._write_root_node()

    def _write_root_node(self) -> None:
        root_dir = self.nodes_dir / "root"
        root_dir.mkdir(parents=True, exist_ok=True)
        if not (root_dir / "proposal.md").exists():
            (root_dir / "proposal.md").write_text(
                f"# Baseline Root (`{self.cycle_id}`)\n\n"
                f"Uncompressed B16 policy (`resident_bytes={B16_RESIDENT_BYTES}`, `score=0.0`).\n",
                encoding="utf-8",
            )
        if not (root_dir / "eval.json").exists():
            (root_dir / "eval.json").write_text(
                json.dumps(
                    {
                        "candidate": "root",
                        "status": "baseline",
                        "resident_bytes": B16_RESIDENT_BYTES,
                        "score": 0.0,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def reset(self) -> None:
        self._revealed.clear()
        self._revealed_order.clear()
        self.best_so_far = 0.0
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0

    def observed(self) -> dict[str, Observation]:
        return dict(self._revealed)

    def legal_roots(self) -> list[str]:
        return [cid for cid in self.roots if cid not in self._revealed]

    def opened_branches(self) -> list[int]:
        seen: list[int] = []
        seen_set: set[int] = set()
        for cid in self._revealed_order:
            b = self._revealed[cid].branch
            if b not in seen_set:
                seen_set.add(b)
                seen.append(b)
        return seen

    def legal_actions(self) -> list[str]:
        actions = self.legal_roots()
        for cid in self._revealed_order:
            for child_id in self.children.get(cid, ()):
                if child_id not in self._revealed:
                    actions.append(child_id)
        return actions

    def meta(self, cell_id: str) -> CellMeta:
        if cell_id not in self._slots:
            raise KeyError(f"unknown cell_id: {cell_id!r}")
        return self._slots[cell_id]

    def _prepare_batch_evidence(self, batch: list[str], round_idx: int) -> Path:
        ev_root = self.cycle_dir / "round_evidence" / f"round_{round_idx:02d}"
        if ev_root.exists():
            shutil.rmtree(ev_root)
        ev_root.mkdir(parents=True, exist_ok=True)

        baseline_doc = {
            "cycle_id": self.cycle_id,
            "round": round_idx,
            "b16_resident_bytes": B16_RESIDENT_BYTES,
            "gates": {"smoke": f">= {self.smoke_gate}/10", "development": f">= {DEV_GATE}/100"},
            "reference_points": _reference_points(self.numbers_path),
        }
        (ev_root / "baseline.json").write_text(
            json.dumps(baseline_doc, indent=2) + "\n", encoding="utf-8"
        )
        (ev_root / "action_space.json").write_text(
            freeform_action_space_doc() + "\n", encoding="utf-8"
        )
        (ev_root / "in_batch_slots.json").write_text(
            json.dumps(
                [
                    {
                        "cell_id": c,
                        "branch": self._slots[c].branch,
                        "attempt": self._slots[c].attempt,
                        "parent_id": self._slots[c].parent_id,
                    }
                    for c in batch
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        obs_dir = ev_root / "observed_nodes"
        obs_dir.mkdir(parents=True, exist_ok=True)
        for cid in self._revealed_order:
            src_node = self.nodes_dir / cid
            dst_node = obs_dir / cid
            dst_node.mkdir(parents=True, exist_ok=True)
            for fname in ("proposal.md", "recipe.json", "eval.json", "error.txt"):
                fpath = src_node / fname
                if fpath.is_file():
                    shutil.copy2(fpath, dst_node / fname)

        prior_dir = ev_root / "prior_cycles"
        prior_dir.mkdir(parents=True, exist_ok=True)
        for pcycle in self.prior_cycle_dirs:
            pnodes = pcycle / "nodes"
            if not pnodes.is_dir():
                continue
            dst_cycle = prior_dir / pcycle.name
            dst_cycle.mkdir(parents=True, exist_ok=True)
            manifest_src = pcycle / "live_cycle_manifest.json"
            if manifest_src.is_file():
                shutil.copy2(manifest_src, dst_cycle / "live_cycle_manifest.json")
            for node_sub in sorted(pnodes.iterdir()):
                if not node_sub.is_dir():
                    continue
                dst_sub = dst_cycle / node_sub.name
                dst_sub.mkdir(parents=True, exist_ok=True)
                for fname in ("proposal.md", "recipe.json", "eval.json", "error.txt"):
                    fpath = node_sub / fname
                    if fpath.is_file():
                        shutil.copy2(fpath, dst_sub / fname)
        return ev_root

    def _expand_single_slot(
        self,
        cid: str,
        batch_ev_root: Path,
        batch_seen_sigs: dict[tuple[Any, ...], str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        node_dir = self.nodes_dir / cid
        ws_dir = node_dir / "workspace"
        logs_dir = node_dir / "logs"
        node_ev_dir = node_dir / "evidence"
        cm = self._slots[cid]

        # Resume safety: if already expanded with eval.json or cached recipe, keep existing artifacts.
        if (node_dir / "eval.json").is_file():
            rec_path = node_dir / "recipe.json"
            if rec_path.is_file():
                recipe = json.loads(rec_path.read_text(encoding="utf-8"))
                self.node_recipes[cid] = recipe
                return recipe, None
            err_txt = (
                (node_dir / "error.txt").read_text(encoding="utf-8")
                if (node_dir / "error.txt").is_file()
                else "prior implementation failure"
            )
            return None, err_txt

        # Inherit parent workspace files into child's workspace / parent_context
        ws_dir.mkdir(parents=True, exist_ok=True)
        if cm.parent_id and cm.parent_id != "root":
            parent_node = self.nodes_dir / cm.parent_id
            parent_ctx = ws_dir / "parent_context"
            if parent_ctx.exists():
                shutil.rmtree(parent_ctx)
            parent_ctx.mkdir(parents=True, exist_ok=True)
            for fname in ("proposal.md", "recipe.json", "eval.json", "error.txt"):
                fpath = parent_node / fname
                if fpath.is_file():
                    shutil.copy2(fpath, parent_ctx / fname)

        # Prepare per-node read-only evidence snapshot
        if node_ev_dir.exists():
            shutil.rmtree(node_ev_dir)
        shutil.copytree(batch_ev_root, node_ev_dir)
        (node_ev_dir / "TARGET_SLOT.json").write_text(
            json.dumps(
                {
                    "cell_id": cid,
                    "branch": cm.branch,
                    "attempt": cm.attempt,
                    "parent_id": cm.parent_id,
                    "vocab_mode": self.vocab_mode,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        base_prompt = _build_discovery_prompt(
            cell_id=cid,
            branch=cm.branch,
            attempt=cm.attempt,
            parent_id=cm.parent_id or "root",
            vocab_mode=self.vocab_mode,
            direction_guidance=self.direction_guidance,
        )

        last_err: str | None = None
        for attempt_idx in (1, 2):
            prompt = (
                base_prompt
                if attempt_idx == 1
                else (
                    f"{base_prompt}\n\n### Previous validation error\n{last_err}\n\n"
                    "Fix `/tmp/work/proposal.md` and `/tmp/work/recipe.json` now."
                )
            )
            res = run_sandboxed_agent(
                prompt_text=prompt,
                workspace_dir=ws_dir,
                evidence_dir=node_ev_dir,
                logs_dir=logs_dir / f"expand_attempt_{attempt_idx}",
                agent_cmd=self.agent_cmd,
                model=self.model,
                tools="read,write,edit,bash",
                thinking=self.thinking,
                timeout=self.agent_timeout,
                use_bwrap=self.use_bwrap,
            )
            if res.returncode != 0:
                last_err = f"Discovery agent exited {res.returncode}: {res.stderr[-300:]}"
                continue

            proposal_path = ws_dir / "proposal.md"
            recipe_path = ws_dir / "recipe.json"
            if not proposal_path.is_file() or not proposal_path.read_text(encoding="utf-8").strip():
                last_err = "Missing or empty /tmp/work/proposal.md"
                continue
            if not recipe_path.is_file():
                last_err = "Missing /tmp/work/recipe.json"
                continue
            try:
                raw_recipe = _extract_json_object(recipe_path.read_text(encoding="utf-8"))
                recipe = validate_freeform_recipe(
                    raw_recipe,
                    expected_id=cid,
                    seen_signatures=batch_seen_sigs,
                    vocab_mode=self.vocab_mode,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                continue

            shutil.copy2(proposal_path, node_dir / "proposal.md")
            (node_dir / "recipe.json").write_text(
                json.dumps(recipe, indent=2) + "\n", encoding="utf-8"
            )
            self.node_recipes[cid] = recipe
            return recipe, None

        (node_dir / "error.txt").write_text(
            (last_err or "expansion failed") + "\n", encoding="utf-8"
        )
        return None, last_err or "expansion failed"

    def _evaluate_slot_gpu(self, cid: str, recipe: dict[str, Any]) -> dict[str, Any]:
        node_dir = self.nodes_dir / cid
        if self.evaluator is not None:
            return self.evaluator(cid, recipe, node_dir)
        for phase_sub in ("smoke", "size"):
            stale = MEASUREMENTS_DIR / cid / phase_sub
            if stale.exists():
                shutil.rmtree(stale)
        for _ in range(3):
            if not gpu_tight():
                break
            time.sleep(120)
        res = run_compression_subprocess(recipe, "smoke", self.recipes_dir)
        failing = _read_smoke_failing_tasks(MEASUREMENTS_DIR / cid / "smoke" / "episodes.csv")
        if failing is not None:
            res["failing_tasks"] = failing
        res["machine"] = gpu_name()
        return res

    def probe_batch(
        self,
        cells: list[str],
        on_reveal: Callable[[Observation], Any] | None = None,
    ) -> list[Observation]:
        if not cells:
            return []
        if len(cells) > self.max_parallelism:
            raise ValueError(
                f"batch size {len(cells)} exceeds max_parallelism={self.max_parallelism}"
            )
        if len(set(cells)) != len(cells):
            raise ValueError("batch contains duplicate cell ids")
        legal = set(self.legal_actions())
        for cid in cells:
            if cid not in legal:
                raise ValueError(f"illegal cell in batch: {cid!r}")

        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(len(cells) / self.max_parallelism)

        batch_ev_root = self._prepare_batch_evidence(cells, self.decision_rounds)
        frozen_sigs = dict(self.seen_signatures)

        # Parallel online discovery expansion against identical frozen prefix
        expanded: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        if self.agent_workers > 1 and len(cells) > 1:
            with ThreadPoolExecutor(max_workers=min(self.agent_workers, len(cells))) as pool:
                futs = {
                    cid: pool.submit(self._expand_single_slot, cid, batch_ev_root, frozen_sigs)
                    for cid in cells
                }
                for cid, fut in futs.items():
                    expanded[cid] = fut.result()
        else:
            for cid in cells:
                expanded[cid] = self._expand_single_slot(cid, batch_ev_root, frozen_sigs)

        # Deduplicate parallel siblings sequentially; retry colliding siblings once with updated seen_sigs
        for cid in cells:
            recipe, err = expanded[cid]
            if recipe is None:
                continue
            sig = canonical_recipe_signature(recipe)
            if sig in self.seen_signatures:
                # Re-expand colliding sibling knowing earlier sibling claimed `sig`
                node_dir = self.nodes_dir / cid
                for artifact in ("recipe.json", "error.txt"):
                    if (node_dir / artifact).exists():
                        (node_dir / artifact).unlink()
                ws_recipe = node_dir / "workspace" / "recipe.json"
                if ws_recipe.exists():
                    ws_recipe.unlink()
                recipe, err = self._expand_single_slot(
                    cid, batch_ev_root, dict(self.seen_signatures)
                )
                expanded[cid] = (recipe, err)
                if recipe is not None:
                    sig = canonical_recipe_signature(recipe)
                    if sig in self.seen_signatures:
                        err = f"Duplicate recipe signature with {self.seen_signatures[sig]}"
                        (node_dir / "error.txt").write_text(err + "\n", encoding="utf-8")
                        expanded[cid] = (None, err)
                        continue
            self.seen_signatures[sig] = cid

        # Sequential physical evaluation (1 GPU) + observation reveal
        out: list[Observation] = []
        for cid in cells:
            cm = self._slots[cid]
            node_dir = self.nodes_dir / cid
            eval_path = node_dir / "eval.json"
            recipe, exp_err = expanded[cid]

            if eval_path.is_file():
                eval_doc = json.loads(eval_path.read_text(encoding="utf-8"))
                err_txt = (
                    (node_dir / "error.txt").read_text(encoding="utf-8").strip()
                    if (node_dir / "error.txt").is_file()
                    else eval_doc.get("error")
                )
            elif recipe is None:
                eval_doc, err_txt = _format_eval_summary(
                    cid=cid,
                    branch=cm.branch,
                    attempt=cm.attempt,
                    parent_id=cm.parent_id or "root",
                    recipe=None,
                    raw_res={"status": "implementation_failure", "error": exp_err},
                    smoke_gate=self.smoke_gate,
                )
                eval_path.write_text(json.dumps(eval_doc, indent=2) + "\n", encoding="utf-8")
                if err_txt:
                    (node_dir / "error.txt").write_text(err_txt + "\n", encoding="utf-8")
            else:
                raw_res = self._evaluate_slot_gpu(cid, recipe)
                eval_doc, err_txt = _format_eval_summary(
                    cid=cid,
                    branch=cm.branch,
                    attempt=cm.attempt,
                    parent_id=cm.parent_id or "root",
                    recipe=recipe,
                    raw_res=raw_res,
                    smoke_gate=self.smoke_gate,
                )
                eval_path.write_text(json.dumps(eval_doc, indent=2) + "\n", encoding="utf-8")
                if err_txt:
                    (node_dir / "error.txt").write_text(err_txt + "\n", encoding="utf-8")

            self.node_evals[cid] = eval_doc
            passed = bool(eval_doc.get("passed_smoke_gate", False))
            score = float(eval_doc.get("score", 0.0))
            fail_class = str(eval_doc.get("fail_class", "runtime_failure"))
            parent_obs = self._revealed.get(cm.parent_id) if cm.parent_id != "root" else None
            delta_parent = (
                score - parent_obs.score
                if parent_obs is not None and parent_obs.score is not None
                else score
            )
            family = (recipe or {}).get("family", f"{self.cycle_id}_failed")
            obs = Observation(
                cell_id=cid,
                branch=cm.branch,
                attempt=cm.attempt,
                parent_id=cm.parent_id,
                score=score,
                evaluated=eval_doc.get("status") == "completed",
                valid=passed,
                fail_class=fail_class,
                error=err_txt,
                delta_vs_baseline=score,
                delta_vs_parent=delta_parent,
                n_valid=int(eval_doc.get("successes", 0)),
                n_total=int(eval_doc.get("episodes", 10)),
                mechanism_family=family,
                summary=(
                    f"{cid}: status={eval_doc.get('status')} "
                    f"smoke={eval_doc.get('successes', 0)}/{eval_doc.get('episodes', 10)} "
                    f"mb={eval_doc.get('resident_mb', 'na')} score={score:.4f}"
                ),
            )
            self._revealed[cid] = obs
            self._revealed_order.append(cid)
            self.budget_spent += 1
            if passed and score > self.best_so_far:
                self.best_so_far = score
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        return out

    def export_tree(self) -> DiscoveryTree:
        """Freeze all revealed nodes of this online cycle into a `DiscoveryTree` for replay."""
        nodes: list[TreeNode] = []
        for idx, cid in enumerate(self._revealed_order):
            obs = self._revealed[cid]
            nodes.append(
                TreeNode(
                    cell_id=cid,
                    branch=obs.branch,
                    attempt=obs.attempt,
                    parent_id=obs.parent_id,
                    seq=idx,
                    score=obs.score,
                    evaluated=obs.evaluated,
                    valid=obs.valid,
                    fail_class=obs.fail_class,
                    error=obs.error,
                    mechanism_family=obs.mechanism_family,
                    summary=obs.summary,
                    cycle=0,
                    successes=obs.n_valid,
                    episodes=obs.n_total,
                )
            )
        return DiscoveryTree(
            world_id=f"dream-v2-{self.cycle_id}",
            baseline_score=self.baseline_score,
            nodes=nodes,
        )


def _recent_manifests(study_dir: Path) -> tuple[dict[str, Any], ...]:
    cycles_dir = study_dir / "cycles"
    if not cycles_dir.is_dir():
        return ()
    out: list[dict[str, Any]] = []
    for sub in sorted(cycles_dir.iterdir()):
        mp = sub / "live_cycle_manifest.json"
        if mp.is_file():
            out.append(json.loads(mp.read_text(encoding="utf-8")))
    return tuple(out)


def _recent_beta_sweeps(study_dir: Path) -> tuple[dict[str, Any], ...]:
    policies_dir = study_dir / "policies"
    if not policies_dir.is_dir():
        return ()
    out: list[dict[str, Any]] = []
    for sub in sorted(policies_dir.iterdir()):
        sp = sub / "evidence" / "beta_sweep.json"
        if sp.is_file():
            out.append(json.loads(sp.read_text(encoding="utf-8")))
    return tuple(out)


def _prior_cycle_dirs_and_sigs(
    study_dir: Path, exclude_cycle_id: str
) -> tuple[list[Path], dict[tuple[Any, ...], str]]:
    cycles_dir = study_dir / "cycles"
    if not cycles_dir.is_dir():
        return [], {}
    dirs: list[Path] = []
    sigs: dict[tuple[Any, ...], str] = {}
    for sub in sorted(cycles_dir.iterdir()):
        if not sub.is_dir() or sub.name == exclude_cycle_id:
            continue
        if (sub / "tree.json").is_file():
            dirs.append(sub)
        nodes_dir = sub / "nodes"
        if nodes_dir.is_dir():
            for nsub in sorted(nodes_dir.iterdir()):
                rpath = nsub / "recipe.json"
                if rpath.is_file():
                    try:
                        rec = json.loads(rpath.read_text(encoding="utf-8"))
                        sigs[canonical_recipe_signature(rec)] = nsub.name
                    except (OSError, json.JSONDecodeError):
                        pass
    return dirs, sigs


def run_online_cycle(
    *,
    cycle_id: str,
    study_dir: Path = DREAM_V2_DIR,
    policy_path: Path | None = None,
    iteration: int = 0,
    probe_budget: int = 8,
    max_parallelism: int = 4,
    hard_max_branches: int = 8,
    hard_max_depth: int = 4,
    promote_top_k: int = 1,
    vocab_mode: str = "any",
    smoke_gate: int = SMOKE_GATE,
    agent_cmd: str | list[str] | None = None,
    model: str | None = None,
    thinking: str | None = None,
    agent_timeout: float = 180.0,
    agent_workers: int = 2,
    use_bwrap: bool = True,
    evaluator: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
    promote_evaluator: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
    direction_guidance: str = "",
) -> dict[str, Any]:
    """Run one complete online Dream-RSI v2 cycle (`plan_grid -> online solve -> promote -> tree`)."""
    study_root = Path(study_dir)
    cycle_dir = study_root / "cycles" / cycle_id
    cycle_dir.mkdir(parents=True, exist_ok=True)

    pol_path = Path(policy_path) if policy_path is not None else INCUMBENT_TEMPLATE
    pol_cls = load_policy_class(pol_path, cycle_id)
    policy = pol_cls({})
    baked_beta = float(getattr(policy, "beta", 0.6))

    manifests = _recent_manifests(study_root)
    sweeps = _recent_beta_sweeps(study_root)
    plan = policy.plan_grid(
        GridPlanningContext(
            iteration=iteration,
            max_parallelism=max_parallelism,
            recent_live_manifests=manifests[-3:],
            recent_beta_sweeps=sweeps[-3:],
            default_budget=probe_budget,
        )
    )
    eff_plan = GridPlan(
        max_branches=max(1, min(hard_max_branches, int(plan.max_branches))),
        max_attempts_per_branch=max(0, min(hard_max_depth, int(plan.max_attempts_per_branch))),
        max_parallelism=max(1, min(max_parallelism, int(plan.max_parallelism))),
        rounds_cap=max(1, int(plan.rounds_cap)),
        rationale=plan.rationale,
    )
    (cycle_dir / "grid_plan.json").write_text(
        json.dumps(asdict(eff_plan), indent=2) + "\n", encoding="utf-8"
    )

    prior_dirs, prior_sigs = _prior_cycle_dirs_and_sigs(study_root, cycle_id)
    question = OnlineDiscoveryQuestion(
        cycle_id=cycle_id,
        grid_plan=eff_plan,
        cycle_dir=cycle_dir,
        evaluator=evaluator,
        prior_cycle_dirs=prior_dirs,
        smoke_gate=smoke_gate,
        vocab_mode=vocab_mode,
        agent_cmd=agent_cmd,
        model=model,
        thinking=thinking,
        agent_timeout=agent_timeout,
        agent_workers=agent_workers,
        use_bwrap=use_bwrap,
        direction_guidance=direction_guidance,
        seen_signatures=prior_sigs,
    )
    sim = policy.solve(question, budget=probe_budget)
    tree = question.export_tree()
    save_tree_json(tree, cycle_dir / "tree.json")

    # Promote top smoke-passing nodes to 100-trial development (never reserved 500-trial final)
    passing = [
        ev
        for ev in question.node_evals.values()
        if ev.get("status") == "completed" and ev.get("passed_smoke_gate")
    ]
    passing.sort(key=lambda r: (r["resident_bytes"], -r["successes"]))
    dev_results: list[dict[str, Any]] = []
    if promote_top_k > 0:
        for ev in passing[:promote_top_k]:
            cid = ev["candidate"]
            recipe = question.node_recipes[cid]
            if promote_evaluator is not None:
                dres = promote_evaluator(cid, recipe, question.nodes_dir / cid)
            else:
                dres = run_compression_subprocess(recipe, "development", question.recipes_dir)
                dres["machine"] = gpu_name()
            if dres.get("status") == "completed" and dres.get("episodes"):
                dres["wilson_95"] = wilson_interval(int(dres["successes"]), int(dres["episodes"]))
            dev_results.append(dres)

    obs_list = [question._revealed[cid] for cid in question._revealed_order]
    deepest_imp = max(
        (o.attempt for o in obs_list if o.valid and o.score and o.score > 0), default=0
    )
    prev_best = max((m.get("best_score", 0.0) for m in manifests), default=0.0)
    plateaued = len(manifests) >= 1 and question.best_so_far <= prev_best

    manifest = {
        "protocol": "dream-rsi-online-cycle-v2",
        "cycle_id": cycle_id,
        "iteration": iteration,
        "policy_path": str(pol_path),
        "baked_in_beta": baked_beta,
        "vocab_mode": vocab_mode,
        "grid_plan": asdict(eff_plan),
        "probes_used": sim.total_probes,
        "decision_rounds": sim.decision_rounds,
        "effective_sequential_rounds": sim.effective_sequential_rounds,
        "opened_branches": question.opened_branches(),
        "best_score": round(question.best_so_far, 6),
        "deepest_improvement": deepest_imp,
        "plateaued": plateaued,
        "world_id": tree.world_id,
        "world_nodes": tree.num_nonroot,
        "smoke_results": list(question.node_evals.values()),
        "development_promotions": dev_results,
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    (cycle_dir / "live_cycle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_dream_rsi_loop(
    *,
    study_dir: Path = DREAM_V2_DIR,
    cycles: int = 3,
    start_iteration: int = 0,
    probe_budget: int = 8,
    max_parallelism: int = 4,
    promote_top_k: int = 1,
    vocab_mode: str = "any",
    m_policy_variants: int = 4,
    agent_cmd: str | list[str] | None = None,
    model: str | None = None,
    thinking: str | None = None,
    agent_timeout: float = 180.0,
    policy_timeout: float = 240.0,
    agent_workers: int = 2,
    use_bwrap: bool = True,
    evaluator: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
    promote_evaluator: Callable[[str, dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run iterative Dream-RSI v2 outer loop: online tree expansion -> LLM policy-code dev -> monotonic update."""
    study_root = Path(study_dir)
    init_dir = study_root / "policies" / "v000_initial"
    init_dir.mkdir(parents=True, exist_ok=True)
    active_policy = init_dir / "policy.py"
    if not active_policy.exists():
        shutil.copy2(INCUMBENT_TEMPLATE, active_policy)

    cycle_summaries: list[dict[str, Any]] = []
    for offset in range(cycles):
        it = start_iteration + offset
        cid = f"iter_{it:03d}"
        manifest = run_online_cycle(
            cycle_id=cid,
            study_dir=study_root,
            policy_path=active_policy,
            iteration=it,
            probe_budget=probe_budget,
            max_parallelism=max_parallelism,
            promote_top_k=promote_top_k,
            vocab_mode=vocab_mode,
            agent_cmd=agent_cmd,
            model=model,
            thinking=thinking,
            agent_timeout=agent_timeout,
            agent_workers=agent_workers,
            use_bwrap=use_bwrap,
            evaluator=evaluator,
            promote_evaluator=promote_evaluator,
        )

        # Load frozen history H_t of all completed online DiscoveryTrees
        trees: list[DiscoveryTree] = []
        for tpath in sorted((study_root / "cycles").glob("iter_*/tree.json")):
            tree = load_tree_json(tpath)
            if tree.num_nonroot > 0:
                trees.append(tree)

        policy_dev_res: dict[str, Any] | None = None
        if trees:
            manifests = list(_recent_manifests(study_root))
            policy_dev_res = develop_policies_llm(
                trees,
                cycle_id=cid,
                incumbent_path=active_policy,
                output_dir=study_root / "policies" / cid,
                live_manifests=manifests,
                m=m_policy_variants,
                max_parallelism=max_parallelism,
                agent_cmd=agent_cmd,
                model=model,
                thinking=thinking,
                timeout=policy_timeout,
                use_bwrap=use_bwrap,
            )
            active_policy = Path(policy_dev_res["selected_policy_path"])

        cycle_summaries.append(
            {
                "cycle_id": cid,
                "best_score": manifest["best_score"],
                "probes_used": manifest["probes_used"],
                "policy_dev": (
                    {
                        "selected_slot": policy_dev_res["selected_slot"],
                        "selected_policy_path": policy_dev_res["selected_policy_path"],
                        "incumbent_V": policy_dev_res["incumbent_in_sample_V"],
                        "selected_V": policy_dev_res["selected_in_sample_V"],
                    }
                    if policy_dev_res
                    else None
                ),
            }
        )

    loop_summary = {
        "protocol": "dream-rsi-loop-v2",
        "study_dir": str(study_root),
        "final_active_policy": str(active_policy),
        "cycles": cycle_summaries,
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    (study_root / "loop_summary.json").write_text(
        json.dumps(loop_summary, indent=2) + "\n", encoding="utf-8"
    )
    return loop_summary
