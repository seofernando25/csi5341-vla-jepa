"""Verify actual CUDA policy loading and prediction; never claim robot success."""

import argparse
import time
from datetime import UTC, datetime

import torch

from evaluation.common import ROOT, environment, write_json
from evaluation.models import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["B16", "Q8", "Q4", "S500"])
    args = parser.parse_args()
    output = ROOT / "studies/evaluation/checks" / f"{args.variant}.json"
    run_id = args.variant + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    history = output.parent / "history" / f"{run_id}.json"
    if output.exists():
        previous = __import__("json").loads(output.read_text())
        previous_path = (
            output.parent
            / "history"
            / (previous.get("run_id", args.variant + "-initial") + ".json")
        )
        if not previous_path.exists():
            write_json(previous_path, previous)
    result = {
        "run_id": run_id,
        "variant": args.variant,
        "kind": "GPU smoke test",
        "environment": environment(),
        "input": "synthetic zeros; not an efficiency benchmark or robot evaluation",
    }
    try:
        started = time.perf_counter()
        policy, metadata = load_policy(args.variant)
        torch.cuda.synchronize()
        result.update(metadata, load_seconds=time.perf_counter() - started)
        batch = {
            "observation.images.image": torch.zeros(1, 3, 224, 224, device="cuda"),
            "observation.images.image2": torch.zeros(1, 3, 224, 224, device="cuda"),
            "observation.state": torch.zeros(1, 8, device="cuda"),
            "task": ["pick up the black bowl and place it on the plate"],
        }
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        assert torch.isfinite(action).all(), "Nonfinite actions"
        assert list(action.shape) == [1, policy.config.chunk_size, 7]
        result.update(
            outcome="passed",
            action_shape=list(action.shape),
            action_range=[action.min().item(), action.max().item()],
        )
    except Exception as exc:
        result.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(history, result)
        write_json(output, result)
        raise
    write_json(history, result)
    write_json(output, result)
    print(f"{args.variant}: passed CUDA smoke; evidence: {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
