"""CLI entrypoint for the Dream-RSI clean-room cycle and policy development."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rsi.agent_backends import BACKENDS, STUDY_DIR, AgentSpec, load_agent_spec
from rsi.policy_dev import DEFAULT_POLICY_TIMEOUT_S

Evaluator = Callable[[Path, str, Path], dict[str, Any]]
EVALUATOR_HELP = (
    "MODULE:FUNC replacing the sandboxed evaluator; called as FUNC(snapshot_dir, phase, out_dir)"
)


def resolve_callable(spec: str | None) -> Evaluator | None:
    """Import `MODULE:FUNC` and return the callable, or None when no spec is given."""
    if spec is None:
        return None
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise SystemExit(f"expected MODULE:FUNC, got {spec!r}")
    func = getattr(importlib.import_module(module_name), func_name, None)
    if not callable(func):
        raise SystemExit(f"{spec!r} is not a callable")
    return func


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-cmd", default=None, help="Override the agent command (tests)")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="Agent backend (default: protocol dream_rsi.agent.backend)",
    )
    parser.add_argument("--model", default=None, help="Model id (default: protocol)")
    parser.add_argument("--effort", default=None, help="Effort level (default: protocol)")


def _agent_spec(args: argparse.Namespace, study_dir: Path) -> AgentSpec:
    return load_agent_spec(study_dir, backend=args.backend, model=args.model, effort=args.effort)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rsi", description="Dream-RSI CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cyc = sub.add_parser("cycle", help="Run one Dream-RSI online per-node discovery cycle")
    cyc.add_argument("--cycle-id", default="iter_000")
    cyc.add_argument("--study-dir", default="studies/program-v1")
    cyc.add_argument("--policy", default=None, help="Path to active OptimalPolicy python file")
    cyc.add_argument("--iteration", type=int, default=0)
    cyc.add_argument("--probe-budget", type=int, default=8)
    cyc.add_argument("--max-parallelism", type=int, default=4)
    cyc.add_argument("--promote-top-k", type=int, default=1)
    _add_agent_args(cyc)
    cyc.add_argument(
        "--agent-timeout", type=float, default=None, help="Seconds (default: protocol)"
    )
    cyc.add_argument("--eval-timeout", type=float, default=None, help="Seconds (default: protocol)")
    cyc.add_argument("--no-bwrap", action="store_true")
    cyc.add_argument("--evaluator", default=None, help=EVALUATOR_HELP)

    dev = sub.add_parser("develop", help="Run LLM policy-code rewriting over frozen DiscoveryTrees")
    dev.add_argument("--cycle-id", default="iter_000")
    dev.add_argument("--study-dir", default="studies/program-v1")
    dev.add_argument(
        "--incumbent", default=None, help="Incumbent policy.py (default rsi/policies/optimal.py)"
    )
    dev.add_argument("--output-dir", default=None)
    dev.add_argument("--m", type=int, default=4)
    dev.add_argument("--max-parallelism", type=int, default=4)
    _add_agent_args(dev)
    dev.add_argument("--timeout", type=float, default=DEFAULT_POLICY_TIMEOUT_S)
    dev.add_argument("--no-bwrap", action="store_true")

    loop = sub.add_parser("loop", help="Run full Dream-RSI outer loop (cycle + LLM policy dev)")
    loop.add_argument("--study-dir", default="studies/program-v1")
    loop.add_argument("--cycles", type=int, default=3)
    loop.add_argument("--start-iteration", type=int, default=0)
    loop.add_argument("--probe-budget", type=int, default=8)
    loop.add_argument("--max-parallelism", type=int, default=4)
    loop.add_argument("--promote-top-k", type=int, default=1)
    loop.add_argument("--m-variants", type=int, default=4)
    _add_agent_args(loop)
    loop.add_argument(
        "--agent-timeout", type=float, default=None, help="Seconds (default: protocol)"
    )
    loop.add_argument(
        "--eval-timeout", type=float, default=None, help="Seconds (default: protocol)"
    )
    loop.add_argument("--policy-timeout", type=float, default=DEFAULT_POLICY_TIMEOUT_S)
    loop.add_argument("--no-bwrap", action="store_true")
    loop.add_argument("--evaluator", default=None, help=EVALUATOR_HELP)
    loop.add_argument("--max-hours", type=float, default=None, help="Wall-clock budget per call")
    loop.add_argument("--max-agent-usd", type=float, default=None, help="Agent cost budget")

    start = sub.add_parser("start", help="Launch the loop detached (systemd user unit)")
    start.add_argument("--study-dir", required=True)
    start.add_argument("loop_args", nargs=argparse.REMAINDER, help="Flags passed to `loop`")

    stat = sub.add_parser("status", help="Show a study's loop progress and log tail")
    stat.add_argument("--study-dir", required=True)
    stat.add_argument("--json", action="store_true")

    wat = sub.add_parser("watch", help="Serve a read-only live page for a study (loopback)")
    wat.add_argument("--study-dir", required=True)
    wat.add_argument("--port", type=int, default=8787)
    wat.add_argument("--host", default="127.0.0.1")

    stp = sub.add_parser("stop", help="Touch STOP, wait for a clean exit, then stop the unit")
    stp.add_argument("--study-dir", required=True)
    stp.add_argument("--grace", type=float, default=120.0, help="Seconds to wait for exit")

    cal = sub.add_parser(
        "calibrate", help="Score the policy-update instructions on synthetic worlds (no GPU)"
    )
    cal.add_argument("--tag", required=True)
    cal.add_argument("--prompt", default=None, help="Policy-update instruction file")
    cal.add_argument("--m", type=int, default=4)
    cal.add_argument("--max-parallelism", type=int, default=2)
    cal.add_argument("--seeds-per-family", type=int, default=1)
    cal.add_argument("--families", default=None, help="Comma-separated family names (default all)")
    _add_agent_args(cal)
    cal.add_argument("--timeout", type=float, default=300.0)
    cal.add_argument("--out-root", default="outputs/rsi-calibration")
    cal.add_argument("--no-bwrap", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "calibrate":
        from pathlib import Path

        from rsi.calibrate import format_summary, run_calibration

        families = args.families.split(",") if args.families else None
        out_root = Path(args.out_root)
        payload = run_calibration(
            tag=args.tag,
            prompt_path=Path(args.prompt) if args.prompt else None,
            m=args.m,
            max_parallelism=args.max_parallelism,
            seeds_per_family=args.seeds_per_family,
            families=families,
            agent_cmd=args.agent_cmd,
            agent=_agent_spec(args, STUDY_DIR),
            timeout=args.timeout,
            out_root=out_root,
            use_bwrap=not args.no_bwrap,
        )
        print(format_summary(payload, out_root / args.tag / "calibration.json"))
        return 0
    if args.cmd == "cycle":
        from pathlib import Path

        from rsi.online_cycle import run_online_cycle

        payload = run_online_cycle(
            cycle_id=args.cycle_id,
            study_dir=Path(args.study_dir),
            policy_path=Path(args.policy) if args.policy else None,
            iteration=args.iteration,
            probe_budget=args.probe_budget,
            max_parallelism=args.max_parallelism,
            promote_top_k=args.promote_top_k,
            agent_cmd=args.agent_cmd,
            agent=_agent_spec(args, Path(args.study_dir)),
            agent_timeout=args.agent_timeout,
            eval_timeout=args.eval_timeout,
            use_bwrap=not args.no_bwrap,
            evaluator=resolve_callable(args.evaluator),
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "develop":
        from pathlib import Path

        from rsi.online_cycle import _recent_manifests
        from rsi.policy_dev import develop_policies_llm, load_tree_json

        study_dir = Path(args.study_dir)
        trees = [load_tree_json(p) for p in sorted((study_dir / "cycles").glob("iter_*/tree.json"))]
        trees = [t for t in trees if t.num_nonroot > 0]
        if not trees:
            raise SystemExit(f"No non-empty cycle trees found under {study_dir / 'cycles'}")
        out_dir = (
            Path(args.output_dir) if args.output_dir else study_dir / "policies" / args.cycle_id
        )
        payload = develop_policies_llm(
            trees,
            cycle_id=args.cycle_id,
            incumbent_path=Path(args.incumbent) if args.incumbent else None,
            output_dir=out_dir,
            live_manifests=list(_recent_manifests(study_dir)),
            m=args.m,
            max_parallelism=args.max_parallelism,
            agent_cmd=args.agent_cmd,
            agent=_agent_spec(args, study_dir),
            timeout=args.timeout,
            use_bwrap=not args.no_bwrap,
            record_root=study_dir,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "loop":
        from pathlib import Path

        from rsi.durable_loop import run_dream_rsi_loop
        from rsi.service import LoopLockedError

        try:
            payload = run_dream_rsi_loop(
                study_dir=Path(args.study_dir),
                cycles=args.cycles,
                start_iteration=args.start_iteration,
                probe_budget=args.probe_budget,
                max_parallelism=args.max_parallelism,
                promote_top_k=args.promote_top_k,
                m_policy_variants=args.m_variants,
                agent_cmd=args.agent_cmd,
                agent=_agent_spec(args, Path(args.study_dir)),
                agent_timeout=args.agent_timeout,
                eval_timeout=args.eval_timeout,
                policy_timeout=args.policy_timeout,
                use_bwrap=not args.no_bwrap,
                evaluator=resolve_callable(args.evaluator),
                max_hours=args.max_hours,
                max_agent_usd=args.max_agent_usd,
            )
        except LoopLockedError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "watch":
        from rsi.watch import run_watch

        return run_watch(args.study_dir, host=args.host, port=args.port)
    if args.cmd in ("start", "status", "stop"):
        return _service_command(parser, args)
    return 1


def _service_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Dispatch `start`, `status` and `stop` to the detached-run service."""
    from rsi import service

    study = Path(args.study_dir)
    if args.cmd == "start":
        loop_args = list(args.loop_args)
        if loop_args[:1] == ["--"]:
            loop_args = loop_args[1:]
        parser.parse_args(["loop", "--study-dir", str(study), *loop_args])  # validate early
        try:
            record = service.start(study, loop_args)
        except (service.LoopLockedError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(record))
        return 0
    if args.cmd == "status":
        payload = service.status(study)
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        for key, value in payload.items():
            if key != "log_tail":
                print(f"{key}: {value}")
        print("--- loop.log (last 10 lines)")
        print("\n".join(payload["log_tail"]))
        return 0
    result = service.stop(study, grace=args.grace)
    print(json.dumps(result))
    return 0 if result["stopped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
