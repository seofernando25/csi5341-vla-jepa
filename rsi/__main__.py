"""CLI entrypoint for Dream-RSI replay and benchmarking."""

from __future__ import annotations

import argparse
import json

from rsi.dream import run_benchmark
from rsi.qwen_search import run_stage2_live_search


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rsi", description="Dream-RSI CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("benchmark", help="Run Dream-RSI replay & policy benchmark")
    bench.add_argument("--attempts", default="studies/dream-rsi/attempts.jsonl")
    bench.add_argument("--output", default="studies/dream-rsi/dream_benchmark.json")
    bench.add_argument("--json", action="store_true", help="Print full JSON output")

    s2 = sub.add_parser("qwen-stage2", help="Run Stage 2 live Qwen boundary search")
    s2.add_argument("--smoke-budget", type=int, default=10)
    s2.add_argument("--promote-top-k", type=int, default=1)
    s2.add_argument("--beta", type=float, default=0.6)
    s2.add_argument("--output", default="studies/dream-rsi/qwen_stage2.json")

    cyc = sub.add_parser("cycle", help="Run one Dream-RSI v2 online per-node discovery cycle")
    cyc.add_argument("--cycle-id", default="iter_000")
    cyc.add_argument("--study-dir", default="studies/dream-rsi-v2")
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

    dev = sub.add_parser(
        "develop", help="Run LLM policy-code rewriting over frozen v2 DiscoveryTrees"
    )
    dev.add_argument("--cycle-id", default="iter_000")
    dev.add_argument("--study-dir", default="studies/dream-rsi-v2")
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

    loop = sub.add_parser("loop", help="Run full Dream-RSI v2 outer loop (cycle + LLM policy dev)")
    loop.add_argument("--study-dir", default="studies/dream-rsi-v2")
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
    if args.cmd == "benchmark":
        payload = run_benchmark(attempts_path=args.attempts, output_path=args.output)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Dream-RSI benchmark completed in {payload['elapsed_ms']} ms")
            inc = payload["incumbent"]
            print("\nIn-sample replay score V (selection and score on the same worlds)")
            for suite_name, suite_data in payload["in_sample"].items():
                print(f"[{suite_name}]")
                for w_key, row in suite_data.items():
                    sel = row["selected_policy"]
                    print(
                        f"  {w_key}: {inc} V={row['policies'][inc]['reward']:.4f} | "
                        f"selected={sel} V={row['policies'][sel]['reward']:.4f} "
                        f"(dV={row['delta_reward']:+.4f})"
                    )
            print("\nLeave-one-world-out replay score V (mean over held-out worlds)")
            for suite_name, suite_data in payload["held_out"].items():
                print(f"[{suite_name}]")
                for w_key, row in suite_data.items():
                    print(
                        f"  {w_key}: {inc} V={row['mean_incumbent_reward']:.4f} | "
                        f"selected V={row['mean_selected_reward']:.4f} | selected beats/below "
                        f"incumbent in {row['folds_selected_beats_incumbent']}/"
                        f"{row['folds_selected_below_incumbent']} of {len(row['folds'])} folds"
                    )
        return 0
    if args.cmd == "qwen-stage2":
        from pathlib import Path

        payload = run_stage2_live_search(
            smoke_budget=args.smoke_budget,
            promote_top_k=args.promote_top_k,
            beta=args.beta,
            output_path=Path(args.output),
        )
        print(json.dumps(payload, indent=2))
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
