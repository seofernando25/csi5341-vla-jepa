# Dream-RSI search (reopened: fidelity repair)

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

Completed Dream-RSI integration under `rsi/` following arXiv:2609.14858v2 (Eq. 1,
Appendix D.2 Listing 2; the paper states no numbered algorithm):

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

### Fidelity repair: what the replay scores can and cannot say

- Scores are Eq. (1) replay values $V$, not LIBERO success. Worlds are not independent: the
  three SmolVLM cycles split one search run, the Qwen chains are synthesized from a grid study,
  and the synthetic worlds are hand-made. The policy-development agent ([develop.py](../../rsi/develop.py))
  archives $M=4$ `OptimalPolicy` revisions per round from replay trajectories (tuned default
  $\beta$, serial/wide arms, gate-retry ablation) with provenance under
  `rsi/policies/revisions/`; selection is argmax including the incumbent on held-out $V$.
- In-sample $V$ (selection and score on the same worlds) overstates rank stability.
  Leave-one-world-out $V$ selects on the other worlds of the suite and scores the held-out
  world. Smoke and development never share a $V$; the development world is in-sample only
  (one world cannot leave-one-out).
- Held-out ($\lambda=0.05$, $W=4$): `real_smoke_derived` (5 worlds) mean selected $V=0.5662$
  vs incumbent $0.3421$, selected beats incumbent in 5/5 folds, always
  `OptimalPolicy[no-gate-retry]`; `synthetic` (4 worlds) $0.4456$ vs $0.2855$, 4/4 folds,
  always `OptimalPolicy`. Flips: on held-out `qwen-compression-smoke` the best replay is
  `RandomPolicy[seed=2]`, not the selection; at $W=1$ synthetic folds split 2/2 and
  `synth-late-bloomer` favors the incumbent. Single-suite in-sample $V$ on
  `qwen_compression_smoke` ($W\ge 4$) picks `RandomPolicy[seed=*]` seed luck; pooled and
  held-out selection picks an `OptimalPolicy` variant.
- Sensitivity ($\lambda \in \{0,0.01,0.05,0.1,0.2\} \times W \in \{1,2,4,8\}$):
  selection is stable across the grid (`OptimalPolicy` on synthetic,
  `OptimalPolicy[no-gate-retry]` on real smoke; one $W=1$ fold flips to `OptimalPolicy`).
  Each cell in `dream_benchmark.json` records hypervolume and parallelism penalty separately.
- Pool: `InitialPolicy` incumbent (argmax ties resolve to it) vs `OptimalPolicy`,
  `OptimalPolicy[no-gate-retry]`, greedy-serial, round-robin, three fixed-seed random
  policies, plus the archived per-round revisions. Replay sweeps beta
  ($\rho \in \{0,0.25,0.5,0.75,1\}$); live rounds fix beta at the trajectory-tuned value.

## Live rounds (RTX 3090)

Frozen `InitialPolicy` control vs the held-out-selected `OptimalPolicy` variant
(`beta=1.0`, `gate_retry=False`, i.e. the `r1/r2-retry-flip` config: LOWO parent at tuned
beta) on fresh grids, same probe budget, independent journals ([live.py](../../rsi/live.py)).
Pre-stated gates: $\ge 8/10$ smoke promotes the smallest passing cell to development (100
trials); development gate $\ge 77/100$. Stage journals record phase, $n$, bytes and machine
(`outputs/compression/r*/`, not committed); compact evidence in
[live_round1.json](live_round1.json) and [live_round2.json](live_round2.json) with committed
measurement `run.json` files. Bases are the restricted-vocabulary `T1`/`T2`, not report C1/C2.

- **Round 1** (9 fresh cells around the Stage 2 winner, budget 6 each): control probed 2
  cells, 0 passes (`6/10`, `7/10`); selected probed 5, 3 passes on the new `group_size=32`
  branch (`r1-t10-v25-g32`, `r1-t15-v35-g32`, `r1-t20-v45-g32`, each `8/10`, Wilson 95%
  `[49.0, 94.3]`). Promoted `r1-t20-v45-g32` scored **83/100** (Wilson 95% `[74.5, 89.1]`)
  at **853.1 MB**, beating the `77/100` gate and saving `124.7 MB` vs `T2` (`977.8 MB`).
- **Round 2** (6 fresh cells tightening the `g32` winner, budget 4 each): both arms probed
  the 2 roots, 0 passes (`6/10`, `6/10`); no promotion. Tighter cuts fail at smoke, so the
  round stopped after 2 probes per arm.
- **Totals:** 7 new smoke runs + 1 development promotion on `NVIDIA GeForce RTX 3090`
  (plus 1 pre-flight smoke); reserved 500 trials untouched; phases never pooled. Replay $V$
  is not LIBERO success, and one `83/100` promotion does not claim a general selection gain.

## Stage 2: Live Qwen boundary search (RTX 5060 Ti)

Using `OptimalPolicy` (`W=4`, $\beta=1.0$) in [qwen_search.py](../../rsi/qwen_search.py) over 5
mixed-precision/pruning branches between the restricted-vocabulary bases `T1` (`838.5 MB`) and
`T2` (`977.8 MB`) (Study B recipes `deploy-nf4-vision-text-channels-restricted` and
`deploy-hqq2-vision-action-restricted`; not the report's C1/C2):

- **Smoke phase (8 probes, 2 decision rounds, 10 trials each):** 3 candidates passed the
  $\ge 8/10$ gate (`s2-c2-t10-v25`: `8/10`, `904.8 MB`; `s2-c2-tlay2`: `8/10`, `925.5 MB`;
  `s2-c2-vmlp25`: `8/10`, `961.6 MB`). Tighter cuts (`s2-c2-t15-v35`, `s2-c2-tmlp10`,
  `s2-c1-hqq2-act`, `s2-c2-tlay4` at `7/10`; `s2-c2-vmlp50` at `6/10`) were pruned.
- **Development promotion (100 trials, `NVIDIA GeForce RTX 5060 Ti`, 402.7 s):**
  `s2-c2-t10-v25` (`T2` + `10%` `text_mlp` + `25%` `vision_mlp` pruning) scored **85/100 (85.0%)**
  with **904.8 MB** (`904,848,818` bytes, **−83.64%** vs. `B16`), beating the `77/100` gate,
  `B16` (`82/100`), and `T1` (`82/100`), and saving `73.0 MB` vs. `T2` (`977.8 MB`, `87/100`).
- **Evidence:** [qwen_stage2.json](qwen_stage2.json). Reserved 500 trials remain untouched.
- **Caveats:** the live journal probed 8 of 15 grid cells, so the committed
  `s2-c2-tmlp20` smoke measurement (`8/10`, `863.8 MB` by linear derivation, absent from the
  journal) never faced the promotion gate; its `run.json` summary phase field reads
  `development` but the run holds 10 episodes under `smoke/`, so the replay treats it as a
  10-trial smoke node. The replay `qwen-stage2-smoke` world therefore has 9 nodes, all
  10-trial smoke; no development measurements enter it.
