# Local hardware reference

All RTX 3090 results ran on this machine; RTX 5090 runs used rented cloud instances. Captured 2026-09-24 ([machine-readable](evaluation/hardware/local_rtx3090.json)); it does not record clocks, power or temperature during runs.

| Component | Recorded configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090; 24 GiB advertised VRAM |
| Architecture | Ampere GA102; compute capability **8.6 / sm_86**; 82 active SMs |
| CPU | AMD Ryzen 7 5700X; 8 physical cores / 16 logical CPUs |
| RAM | 67,336,445,952 OS-visible bytes (62.7 GiB) |
| Motherboard | Gigabyte B450 I AORUS PRO WIFI-CF |
| Storage | WD Blue SA510 4 TB SATA SSD; Crucial CT1000P1SSD8 1 TB NVMe SSD |
| OS / kernel | CachyOS Linux; 7.2.6-1-cachyos; x86-64 |
| Driver / runtime | NVIDIA 580.178.04; PyTorch 2.11.0+cu128, CUDA runtime 12.8 |
| Host link | Reported current/maximum PCIe **Gen 3 ×16** in this system |
| Power configuration | Reported current/default cap **420 W**, configurable range 100–450 W |
| CPU threads | PyTorch reports 8 intra-op threads at capture |

## Relevant GPU capabilities

- Ampere GA102, compute capability 8.6: native BF16, FP16, TF32 and INT8/INT4 tensor-core modes.
  No native FP8/FP4: NF4 stores 4-bit weights but computes in BF16.
- Kernel support decides speed. At batch size 1, every quantized configuration in this project
  is slower than BF16 (see the [report](../report/main.pdf)).
- Hardware structured sparsity exists; only packed 2:4 recipes in the compression study use it.

## Capture another machine

```bash
python -m evaluation.hardware --label cloud-GPU-NAME --output outputs/cloud/hardware.json
```

The collector records CPU allocation, cgroup limits and shared memory, and excludes hostnames,
IPs, GPU UUIDs, serial numbers, credentials and mount paths.
