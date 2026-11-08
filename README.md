# How Small Can VLA-JEPA Get?

CSI 5341 course project by Noah Sprenger and Fernando Nogueira. We compress the released
[VLA-JEPA](https://arxiv.org/abs/2602.10098) LIBERO policy after training and measure what
it costs in task success, latency and GPU memory.

**[Final report (PDF)](report/main.pdf)** · [Report source](report/) · [Studies and evidence](studies/README.md) · [Paper presentation](web/presentation/README.md)

## Results

LIBERO-Spatial, 500 reserved trials per configuration, one RTX 3090, batch size 1. Changes are
paired against B16 on identical initial states and seeds.

| Configuration | Success | Δ vs B16, pp [95% CI] | Peak memory | Median latency |
|---|---:|---:|---:|---:|
| B16: released BF16 policy | 95.0% | — | 5.61 GB | 79 ms |
| Q8: LLM.int8 language decoder | 94.8% | −0.2 [−1.2, +0.8] | 4.30 GB (−23%) | 157 ms |
| Q4: NF4 language decoder | 95.8% | +0.8 [−0.6, +2.2] | 3.54 GB (−37%) | 94 ms |
| C1: NF4 everywhere + MLP channel pruning¹ | 93.4% | −1.6 [−3.2, 0.0] | 0.94 GB (−83%) | 135 ms |
| C2: NF4 language, 2-bit HQQ vision/action¹ | 98.0% | +3.0 [+1.2, +5.0] | 1.07 GB (−81%) | 144 ms |
| C3: C2 with full vocabulary | 96.8% | +1.8 [−0.2, +3.8] | 1.38 GB (−75%) | 143 ms |
| C4: C1 + 1 flow step, CUDA-graph engine, dense weights² | 95.4% | +0.4 [−1.8, +2.6] | 4.28 GB (−24%) | 20 ms |
| C5: C4 with packed weights² | 95.2% | +0.2 [−1.8, +2.2] | 1.02 GB (−82%) | 29 ms |
| S500: SmolVLM2-500M backbone, fine-tuned | 2.2% | −92.8 | 3.30 GB (−41%) | 1270 ms |

¹ Embedding table restricted to the 53 token IDs in LIBERO prompts.
² Peak is allocated memory (the allocator reserved 4.50 and 2.55 GB). Dense weights keep NF4-rounded
bf16 copies beside the packed ones.

- NF4 decoder quantization meets the proposal target (≥25% less memory, ≤5 points lost). LLM.int8
  misses it: 25.1% is its theoretical ceiling.
- The language decoder limits compression. Vision and action weights tolerate 2-bit quantization.
- Under the stock execution path every compressed policy is slower at batch size 1: it is launch-bound.
  A CUDA-graph engine (`evaluation/fastpath.py`) runs C1's recipe in 20 ms (4.28 GB) or 29 ms (1.02 GB).
- Offline action change on fixed frames screens out bad recipes (Spearman ρ = −0.67 over 163)
  but cannot rank near-lossless ones.
- Replacing Qwen with SmolVLM2-500M failed within our training budget.

## Layout

| Path | Contents |
|---|---|
| `report/` | Final IEEE report: `main.tex`, `plots/` (all figures from committed evidence), `data/numbers.json` |
| `studies/` | Compact evidence for every experiment: run records, per-episode CSVs, milestones |
| `evaluation/` | Rollouts, timing, S500 training, analysis; compression recipes and transforms in `evaluation/compression/` |
| `src/lerobot_policy_vla_jepa_smolvlm/` | LeRobot plugin for the SmolVLM2 backbone (S500) |
| `web/presentation/` | Paper presentation (October milestone) |
| `tests/` | Unit tests for statistics, compression transforms and the plugin |

## Setup

Linux with an NVIDIA GPU for rollouts and timing; any machine for tests and the report.

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval --extra report
uv run --no-sync pytest
uv run --no-sync python -m report.plots     # rebuild figures and numbers.json
cd report && latexmk -pdf main.tex           # rebuild the PDF
```

Run `uv sync` with the extras above; a plain `uv run` re-syncs and drops them.
Reproduction commands for each study are in its README under [studies/](studies/README.md).

LeRobot is pinned to `30074f7f1358b3c015ae1750017200e86e9c4eb6`. No weights or datasets are
redistributed; see [NOTICE.md](NOTICE.md).

References: [VLA-JEPA](https://arxiv.org/abs/2602.10098), [V-JEPA 2](https://arxiv.org/abs/2506.09985),
[SmolVLM](https://arxiv.org/abs/2504.05299), [LIBERO](https://arxiv.org/abs/2306.03310),
[Dream-RSI](https://arxiv.org/abs/2609.14858).
