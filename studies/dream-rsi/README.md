# Dream-RSI search (closed)

Optional arm: an automated search over SmolVLM conditioning architectures, scored by held-out
prediction loss. It found lower loss but no successful policy.

- GPU study: 23 measured 500-step screens and six 1,500-step promotions. Best candidate n0008
  (residual adapter before the final decoder layer, backbone frozen) reached promotion loss
  0.2762 versus 0.3197 for the matched SmolVLM baseline.
- Its [longer confirmation](../confirmation/n0008-20261001/README.md) scored 0/10 development trials.
- The GPU pipeline rescaled images twice, so the ranking describes a degraded visual input
  ([recovery](../recovery/README.md)). Loss here is not LIBERO success.

[attempts.jsonl](attempts.jsonl) is the exported search journal; [figures/](figures/) holds the
screen, promotion and loss/latency plots. The earlier search harness was archived at commit `82df05d`.

## Dream-RSI integration and replay benchmark

Completed Dream-RSI integration under `rsi/` following arXiv:2609.14858v2 (Algorithm 1, Eq. 1,
Appendix D.2 Listing 2):

- [api.py](../../rsi/api.py): prefix-only `Observation`, `CellMeta`, `GridPlanningContext`,
  `GridPlan`, `SimResult`, and `LLMDesignedMethod` interface.
- [tree.py](../../rsi/tree.py): `DiscoveryTree` and deterministic `ReplayQuestion` (`Child`
  reveal, legality, and batch parallelism checks).
- [objective.py](../../rsi/objective.py): Eq. 1 normalized Pareto hypervolume minus parallelism
  penalty ($\lambda=0.05$) across $\rho \in \{0, 0.25, 0.5, 0.75, 1\}$.
- [initial.py](../../rsi/policies/initial.py) and [optimal.py](../../rsi/policies/optimal.py):
  serial `InitialPolicy` ($\pi_0$) vs. dynamic-portfolio `OptimalPolicy` ($\pi_1$) with
  trajectory reconstruction, repairable-failure classification, `_schedule(beta)`, and `plan_grid`.
- [dream.py](../../rsi/dream.py): `DreamEngine` monotonic selection ($\arg\max_m V^m$) and
  benchmark runner.

Reproduce: `uv run --no-sync python -m rsi benchmark` (exports [dream_benchmark.json](dream_benchmark.json)).

| Suite | $W$ | Initial $V$ | Optimal $V$ | Optimal penalty | Selected | $\Delta V$ |
|---|---:|---:|---:|---:|---|---:|
| `smolvlm_full` (27 nodes) | 1 | 0.8759 | 0.8759 | 1.000 | `InitialPolicy` | +0.0000 |
| `smolvlm_full` (27 nodes) | 2 | 0.8759 | 0.8900 | 0.718 | `OptimalPolicy` | +0.0141 |
| `smolvlm_full` (27 nodes) | 4 | 0.8759 | 0.8931 | 0.656 | `OptimalPolicy` | +0.0172 |
| `smolvlm_cycles` (3 worlds) | 1 | 0.4750 | 0.6299 | 1.000 | `OptimalPolicy` | +0.1549 |
| `smolvlm_cycles` (3 worlds) | 2 | 0.4750 | 0.6461 | 0.675 | `OptimalPolicy` | +0.1711 |
| `smolvlm_cycles` (3 worlds) | 4 | 0.4750 | 0.6496 | 0.605 | `OptimalPolicy` | +0.1746 |
| `synthetic_suite` (4 worlds) | 1 | 0.2855 | 0.2754 | 1.000 | `InitialPolicy` | -0.0101 |
| `synthetic_suite` (4 worlds) | 2 | 0.2855 | 0.4425 | 0.623 | `OptimalPolicy` | +0.1570 |
| `synthetic_suite` (4 worlds) | 4 | 0.2855 | 0.4456 | 0.517 | `OptimalPolicy` | +0.1601 |
| `qwen_compression` (162 smoke + 23 dev) | 1 | 0.1949 | 0.3323 | 1.000 | `OptimalPolicy` | +0.1374 |
| `qwen_compression` (162 smoke + 23 dev) | 2 | 0.1949 | 0.3413 | 0.738 | `OptimalPolicy` | +0.1464 |
| `qwen_compression` (162 smoke + 23 dev) | 4 | 0.1949 | 0.3432 | 0.699 | `OptimalPolicy` | +0.1483 |

## Stage 2: Live Qwen boundary search (RTX 5060 Ti)

Using `OptimalPolicy` (`W=4`, $\beta=1.0$) in [qwen_search.py](../../rsi/qwen_search.py) over 5
mixed-precision/pruning branches between `C1` (`838.5 MB`) and `C2` (`977.8 MB`):

- **Smoke phase (8 probes, 2 decision rounds, 10 trials each):** 3 candidates passed the
  $\ge 8/10$ gate (`s2-c2-t10-v25`: `8/10`, `904.8 MB`; `s2-c2-tlay2`: `8/10`, `925.5 MB`;
  `s2-c2-vmlp25`: `8/10`, `961.6 MB`). Tighter cuts (`s2-c2-t15-v35`, `s2-c2-tmlp10`,
  `s2-c1-hqq2-act`, `s2-c2-tlay4` at `7/10`; `s2-c2-vmlp50` at `6/10`) were pruned.
- **Development promotion (100 trials, `NVIDIA GeForce RTX 5060 Ti`, 402.7 s):**
  `s2-c2-t10-v25` (`C2` + `10%` `text_mlp` + `25%` `vision_mlp` pruning) scored **85/100 (85.0%)**
  with **904.8 MB** (`904,848,818` bytes, **−83.64%** vs. `B16`), beating the `77/100` gate,
  `B16` (`82/100`), and `C1` (`82/100`), and saving `73.0 MB` vs. `C2` (`977.8 MB`, `87/100`).
- **Evidence:** [qwen_stage2.json](qwen_stage2.json). Reserved 500 trials remain untouched.
