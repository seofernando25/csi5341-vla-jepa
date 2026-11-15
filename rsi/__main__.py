"""CLI entrypoint for the Dream-RSI clean-room cycle and policy development."""

from __future__ import annotations

import argparse
import json


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
    cyc.add_argument("--vocab-mode", choices=("any", "full", "restricted"), default="any")
    cyc.add_argument("--agent-cmd", default=None, help="Override pi command (tests/custom wrapper)")
    cyc.add_argument(
        "--model", default=None, help="pi model id (default $RSI_PI_MODEL or deepseek-v4.1-flash)"
    )
    cyc.add_argument("--thinking", default=None)
    cyc.add_argument("--agent-timeout", type=float, default=180.0)
    cyc.add_argument("--agent-workers", type=int, default=2)
    cyc.add_argument("--no-bwrap", action="store_true")
    cyc.add_argument("--direction", default="")

    dev = sub.add_parser("develop", help="Run LLM policy-code rewriting over frozen DiscoveryTrees")
    dev.add_argument("--cycle-id", default="iter_000")
    dev.add_argument("--study-dir", default="studies/program-v1")
    dev.add_argument(
        "--incumbent", default=None, help="Incumbent policy.py (default rsi/policies/optimal.py)"
    )
    dev.add_argument("--output-dir", default=None)
    dev.add_argument("--m", type=int, default=4)
    dev.add_argument("--max-parallelism", type=int, default=4)
    dev.add_argument("--agent-cmd", default=None)
    dev.add_argument("--model", default=None)
    dev.add_argument("--thinking", default=None)
    dev.add_argument("--timeout", type=float, default=240.0)
    dev.add_argument("--no-bwrap", action="store_true")

    loop = sub.add_parser("loop", help="Run full Dream-RSI outer loop (cycle + LLM policy dev)")
    loop.add_argument("--study-dir", default="studies/program-v1")
    loop.add_argument("--cycles", type=int, default=3)
    loop.add_argument("--start-iteration", type=int, default=0)
    loop.add_argument("--probe-budget", type=int, default=8)
    loop.add_argument("--max-parallelism", type=int, default=4)
    loop.add_argument("--promote-top-k", type=int, default=1)
    loop.add_argument("--vocab-mode", choices=("any", "full", "restricted"), default="any")
    loop.add_argument("--m-variants", type=int, default=4)
    loop.add_argument("--agent-cmd", default=None)
    loop.add_argument("--model", default=None)
    loop.add_argument("--thinking", default=None)
    loop.add_argument("--agent-timeout", type=float, default=180.0)
    loop.add_argument("--policy-timeout", type=float, default=240.0)
    loop.add_argument("--agent-workers", type=int, default=2)
    loop.add_argument("--no-bwrap", action="store_true")

    args = parser.parse_args(argv)
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
            vocab_mode=args.vocab_mode,
            agent_cmd=args.agent_cmd,
            model=args.model,
            thinking=args.thinking,
            agent_timeout=args.agent_timeout,
            agent_workers=args.agent_workers,
            use_bwrap=not args.no_bwrap,
            direction_guidance=args.direction,
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
            model=args.model,
            thinking=args.thinking,
            timeout=args.timeout,
            use_bwrap=not args.no_bwrap,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.cmd == "loop":
        from pathlib import Path

        from rsi.online_cycle import run_dream_rsi_loop

        payload = run_dream_rsi_loop(
            study_dir=Path(args.study_dir),
            cycles=args.cycles,
            start_iteration=args.start_iteration,
            probe_budget=args.probe_budget,
            max_parallelism=args.max_parallelism,
            promote_top_k=args.promote_top_k,
            vocab_mode=args.vocab_mode,
            m_policy_variants=args.m_variants,
            agent_cmd=args.agent_cmd,
            model=args.model,
            thinking=args.thinking,
            agent_timeout=args.agent_timeout,
            policy_timeout=args.policy_timeout,
            agent_workers=args.agent_workers,
            use_bwrap=not args.no_bwrap,
        )
        print(json.dumps(payload, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
