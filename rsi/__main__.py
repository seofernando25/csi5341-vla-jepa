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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
