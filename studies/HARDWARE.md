# Local hardware reference

Captured 2026-09-24 UTC with fastfetch, Linux inventory, `nvidia-smi`, and PyTorch. [Machine-readable inventory](evaluation/hardware/local_rtx3090.json). This records the machine at capture time; it does not reconstruct power, clocks, or temperatures during earlier runs.

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

The card's architectural PCIe capability is Gen 4; the installed host link is Gen 3. The reported power configuration is specific to this card/firmware, not the reference RTX 3090 specification. Neither maximum reported clocks nor power limits are measured sustained operating values.

## GPU capabilities relevant to this study

The RTX 3090 has 10,496 CUDA cores and 328 third-generation Tensor Cores, with GDDR6X memory. These are hardware specifications, not measured application throughput. [NVIDIA GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf).

| Capability | Interpretation for our experiments |
| --- | --- |
| FP32; Tensor Core FP16, BF16, TF32 | Native BF16 support was confirmed by PyTorch. The campaign explicitly disables TF32 where configured; capability does not imply use. |
| Integer Tensor Core INT8 / INT4 | Available hardware modes; actual acceleration depends on kernel selection, shape, and batch size. |
| FP8 / FP4 Tensor Core arithmetic | Not native Ampere modes. Q4's **NF4 weight storage with BF16 computation** is not native FP4 arithmetic. |
| Structured sparsity | Hardware support exists; this study does not introduce sparse weights. |
| FP64 | Limited GA102 throughput; not an FP64-focused accelerator or an experimental arm here. |

Supported matrix formats are documented in the [Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.8.0/ampere-tuning-guide/index.html#improved-tensor-core-operations). The installed PyTorch build advertises `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`. This list alone does not establish compatibility of bitsandbytes or other extensions on a new GPU; run the actual policy kernels there.

Q8 internally casts decoder activations to FP16. Q4 uses NF4 with double quantization and BF16 compute. Vision, action, and world-model modules remain BF16 during inference. Smaller stored weights need not yield lower latency, as the current RTX 3090 measurements demonstrate.

## Capture another machine

```bash
python -m evaluation.hardware --label cloud-GPU-NAME --output outputs/cloud/hardware.json
```

The collector records CPU allocation/cgroup limits and shared-memory capacity, useful for rented containers. It excludes hostnames, IPs, GPU UUIDs, serial numbers, credentials, and mount paths. Preserve each capture separately. The later experiment is specified in [CROSS_HARDWARE_PLAN.md](CROSS_HARDWARE_PLAN.md).
