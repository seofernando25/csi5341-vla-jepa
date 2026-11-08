"""Count kernel launches and host synchronizations in one stock action-chunk prediction.

Example: python -m evaluation.launch_profile B16 Q8 Q4
Profiles 10 predictions on the first bank observation per variant and writes
studies/evaluation/launch_profile.json. GPU-busy time is the summed kernel time; when it is far
below wall time, the stock path is launch-bound rather than compute-bound.
"""

import argparse
import gc
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from evaluation.common import ROOT, write_json
from evaluation.models import load_policy
from evaluation.pipeline import Pipeline, inference_autocast, observation_bank

OUTPUT = ROOT / "studies/evaluation/launch_profile.json"
CALLS = 10


def predict(policy, batch):
    policy.reset()
    torch.manual_seed(0)
    with torch.inference_mode(), inference_autocast():
        return policy.predict_action_chunk(batch)


def profile_variant(variant):
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, _ = load_policy(variant)
    batch = Pipeline(policy).sample(observation_bank()[0])

    for _ in range(15):
        predict(policy, batch)
    wall = []
    for _ in range(30):
        torch.cuda.synchronize()
        started = time.perf_counter()
        predict(policy, batch)
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(CALLS):
            predict(policy, batch)
        torch.cuda.synchronize()
    events = trace.key_averages()
    kernels = [e for e in events if e.device_type.name == "CUDA"]
    count = {e.key: e.count / CALLS for e in events}
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "wall_median_ms": float(np.median(wall)),
        "gpu_busy_ms": sum(e.self_device_time_total for e in kernels) / CALLS / 1000,
        "kernel_launches": sum(e.count for e in kernels) / CALLS,
        "host_syncs": count.get("cudaStreamSynchronize", 0.0),
        "nonzero_calls": count.get("aten::nonzero", 0.0),
        "item_calls": count.get("aten::item", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", choices=["B16", "Q8", "Q4"])
    args = parser.parse_args()
    result = {
        "observation": "bank index 0",
        "profiled_calls": CALLS,
        "variants": {v: profile_variant(v) for v in args.variants},
    }
    write_json(OUTPUT, result)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
