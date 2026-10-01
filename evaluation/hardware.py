"""Capture portable hardware provenance without hostnames, IDs, serials or paths."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import write_json


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture(label):
    raw = command("fastfetch", "--format", "json", "--structure", "OS:CPU:Board")
    fetched = {r["type"]: r.get("result", {}) for r in json.loads(raw)} if raw else {}
    cpu = fetched.get("CPU", {})
    if not cpu:
        raw_cpu = command("lscpu", "--json")
        fields = (
            {r["field"].rstrip(":"): r["data"] for r in json.loads(raw_cpu)["lscpu"]}
            if raw_cpu
            else {}
        )
        cpu = {"cpu": fields.get("Model name"), "cores": {"logical": os.cpu_count()}}
    memory = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    limits = {}
    for name in ["cpu.max", "memory.max"]:
        path = Path("/sys/fs/cgroup") / name
        limits[name] = path.read_text().strip() if path.exists() else None
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "compute_cap",
        "power.limit",
        "power.default_limit",
        "power.min_limit",
        "power.max_limit",
        "pcie.link.gen.current",
        "pcie.link.gen.max",
        "pcie.link.width.current",
        "pcie.link.width.max",
        "clocks.max.sm",
        "clocks.max.memory",
    ]
    raw_gpu = command(
        "nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"
    )
    gpus = (
        [
            dict(zip(fields, (v.strip() for v in row), strict=True))
            for row in csv.reader(raw_gpu.splitlines())
        ]
        if raw_gpu
        else []
    )
    raw_disks = command(
        "lsblk", "--json", "--bytes", "--nodeps", "--output", "MODEL,SIZE,ROTA,TRAN,TYPE"
    )
    storage = (
        [
            {k: d.get(k) for k in ["model", "size", "rota", "tran"]}
            for d in json.loads(raw_disks)["blockdevices"]
            if d.get("type") == "disk" and d.get("model")
        ]
        if raw_disks
        else []
    )
    packages = {}
    for name in [
        "torch",
        "transformers",
        "lerobot",
        "hf-libero",
        "bitsandbytes",
        "mujoco",
        "robosuite",
        "numpy",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_info = {}
    try:
        import torch

        torch_info = {"cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads()}
        if torch.cuda.is_available():
            torch_info.update(
                compiled_architectures=torch.cuda.get_arch_list(),
                bf16_hardware_supported=torch.cuda.is_bf16_supported(including_emulation=False),
                devices=[
                    {
                        "name": p.name,
                        "compute_capability": [p.major, p.minor],
                        "multiprocessors": p.multi_processor_count,
                        "usable_memory_bytes": p.total_memory,
                    }
                    for p in [
                        torch.cuda.get_device_properties(i)
                        for i in range(torch.cuda.device_count())
                    ]
                ],
            )
    except ImportError:
        pass
    board = fetched.get("Board", {})
    shm = os.statvfs("/dev/shm")
    return {
        "label": label,
        "captured_at": datetime.now(UTC).isoformat(),
        "capture_scope": "Inventory at capture time; not retrospective per-run power/thermal telemetry",
        "os": fetched.get("OS", {}).get(
            "prettyName", platform.freedesktop_os_release().get("PRETTY_NAME", platform.system())
        ),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "cpu": {
            "model": cpu.get("cpu"),
            "cores": cpu.get("cores"),
            "allowed_logical_cpus": len(os.sched_getaffinity(0)),
        },
        "board": {"vendor": board.get("vendor"), "model": board.get("name")},
        "ram_os_visible_bytes": memory,
        "cgroup_limits": limits,
        "shared_memory_capacity_bytes": shm.f_blocks * shm.f_frsize,
        "gpus": gpus,
        "gpu_units": {"memory.total": "MiB", "power.*": "W", "clocks.*": "MHz"},
        "storage": storage,
        "storage_size_unit": "bytes",
        "python": platform.python_version(),
        "uv": command("uv", "--version"),
        "fastfetch": command("fastfetch", "--version"),
        "packages": packages,
        "torch": torch_info,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new capture filename; preserve earlier hardware evidence")
    write_json(args.output, capture(args.label))
    print("Hardware inventory exported")


if __name__ == "__main__":
    main()
