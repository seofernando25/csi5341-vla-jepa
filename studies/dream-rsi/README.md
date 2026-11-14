# Dream-RSI search (reopened: fidelity repair)

## Executive summary (2026-11-13)

One-line state: four live compression points (83–88/100 dev, −89 to −125 MB) at
indistinguishable success; replay $V$ selected the search policy but is not LIBERO success.

**What stuck.**

- Full-vocab C1 base (Round 4): moved off restricted-vocab `T1`/`T2` onto the report's
  `C1` base (`int8_embedding`). All 5 probed `r4-*` cells passed smoke (8/10);
  `r4-c1-t15-v35-g32` scored **86/100 dev** (Wilson 95% [77.9, 91.5]) at **1199.6 MB**
  (−88.8 MB vs C1's 1288.4 MB, 87/100 dev). `solver_steps=1` passed smoke 8/10.
- Best restricted-vocab (`T2`) points: Stage 2 `s2-c2-t10-v25` **85/100 dev** (904.8 MB,
  RTX 5060 Ti); R1 `r1-t20-v45-g32` **83/100 dev** (853.1 MB, −124.7 MB vs T2, RTX 3090);
  R3 `r3-t15-v40-g32` **88/100 dev** [80.2, 93.0] (885.1 MB, −92.7 MB vs T2);
  R5 `r5-hqq3-t20-v40` (HQQ 3-bit vision) **85/100 dev** [76.7, 90.7] (876.0 MB, −101.8 MB vs T2).
- Method: leave-one-world-out replay over 10 coarse / 22 fine / 5 live / 4 synthetic worlds
  (`OptimalPolicy[no-gate-retry]` wins 10/10 coarse folds, $V=0.3854$ vs $0.2192$, and 5/5
  live folds, $V=0.2047$ vs $0.0964$); $M=4$ automated revision generator
  (`rsi/develop.py`, archived `r1..r6`); sequential multi-fidelity gate (`rsi/gate.py`,
  10-trial good-branch false kills 13.08% to 4.43% at $p=0.87$); parallel LIBERO rollout
  engine (`evaluation/parallel_rollout.py`).

**What did not work.**

- Tighter MLP cuts past `t20-v45` (Round 2): `t20..25`/`v45..50` failed smoke (6/10 both
  arms); pruning cliff at ~820–845 MB.
- Attention-head pruning (Round 5): 12.5% cut (`r5-heads-t15-v35`) dropped to 6/10 smoke.
- HQQ 3-bit vision (Round 5): `r5-hqq3-t20-v40` (85/100 dev) still scored 5/10, 7/10, 4/10
  on tasks 0, 5, 8 — vision bitwidth is not the bottleneck on those tasks.
- Small-grid ties and dev-100 noise: R1 selected beat frozen control (3 passes vs 0); in
  R3–R5 root cells already passed 8/10 so both policies probed the same roots and tied.
  Dev runs of 83–88/100 have overlapping 95% Wilson intervals ($\pm 8$–$9$ pp); the
  deterministic win is weight reduction (−89 to −125 MB) at indistinguishable success.

Details below; older SmolVLM negative results kept intact.

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
- Held-out ($\lambda=0.05$, $W=4$): `real_smoke_coarse` (10 worlds: 3 SmolVLM cycles,
  Qwen smoke, Stage 2 smoke, live R1, R2, R3, R4, R5) mean selected $V=0.3854$ vs incumbent
  $0.2192$, selected beats incumbent in 10/10 folds, unanimous
  `OptimalPolicy[no-gate-retry]`; `real_smoke_fine` (22
  worlds: 6 SmolVLM branch + 3 fragment + 13 Qwen family) $0.1361$ vs $0.0344$, beats
  incumbent in 6/22 folds with 16 ties on single-chain worlds, always `OptimalPolicy`;
  `live_rounds` (R1+R2+R3+R4+R5) $0.2047$ vs $0.0964$, selected beats incumbent in 5/5 folds
  (at $W=1$ 2/5 beat, rest tie); `synthetic` (4 worlds) $0.4456$ vs $0.2855$,
  4/4 folds, always `OptimalPolicy`. Flips: on held-out `qwen-compression-smoke` the
  best replay is `RandomPolicy[seed=2]`, not the selection; at $W=1$ synthetic folds
  split 2/2 and `synth-late-bloomer` favors the incumbent. Single-suite in-sample $V$
  on `qwen_compression_smoke` ($W\ge 4$) picks `RandomPolicy[seed=*]` seed luck; pooled
  and held-out selection picks an `OptimalPolicy` variant.
- Stability (seeds 0–4, $5 \times 4$ $\lambda \times W$ cells per suite): the winner
  survives seeds on every real-smoke suite (unanimous fraction 1.0 on coarse, fine and
  live rounds across all 20 cells). It does not survive serial replay on synthetic
  worlds (0.75): all five $W=1$ cells flip by seed (`RandomPolicy[seed=1]`,
  `RandomPolicy[seed=2]`, `EpsilonGreedyPolicy[seed=4]` beat the incumbent's modal
  win), while every $W \ge 2$ cell is unanimous. Serial replay lets a stochastic draw
  win; that is seed luck, not LIBERO evidence.
- Sensitivity ($\lambda \in \{0,0.01,0.05,0.1,0.2\} \times W \in \{1,2,4,8\}$):
  selection is stable across the grid (`OptimalPolicy` on synthetic and fine smoke,
  `OptimalPolicy[no-gate-retry]` on coarse smoke, `TwoStageGatePolicy` on live
  rounds). Each cell in `dream_benchmark.json` records hypervolume and parallelism
  penalty separately.
- Pool: `InitialPolicy` incumbent (argmax ties resolve to it) vs `OptimalPolicy`,
  `OptimalPolicy[no-gate-retry]`, greedy-serial, round-robin, epsilon-greedy
  ($\epsilon=0.2$), shallow-first (size-penalized argmax proxy: mildest attempt
  first), two-stage gate with one near-miss retry, incumbent-with-exceptions (serial
  plus one near-miss excuse), three fixed-seed random policies, plus the archived
  per-round revisions. Replay sweeps beta
  ($\rho \in \{0,0.25,0.5,0.75,1\}$); live rounds fix beta at the trajectory-tuned
  value. Worlds and their provenance are listed under `world_provenance` in
  [dream_benchmark.json](dream_benchmark.json); each suite is analyzed separately
  and never pooled.

## Live rounds (RTX 3090)

Frozen `InitialPolicy` control vs the held-out-selected `OptimalPolicy` variant
(`beta=1.0`, `gate_retry=False`, i.e. the `r1/r2/r3-retry-flip` config: LOWO parent at tuned
beta) on fresh grids, same probe budget, independent journals ([live.py](../../rsi/live.py)).
Pre-stated gates: $\ge 8/10$ smoke promotes the smallest passing cell to development (100
trials); development gate $\ge 77/100$. Stage journals record phase, $n$, bytes and machine
(`outputs/compression/r*/`, not committed); compact evidence in
[live_round1.json](live_round1.json), [live_round2.json](live_round2.json),
[live_round3.json](live_round3.json), [live_round4.json](live_round4.json) and
[live_round5.json](live_round5.json) with committed
measurement `run.json` files. Rounds 1/2/3/5 use the restricted-vocabulary `T1`/`T2`
bases; Round 4 uses the report's full-vocabulary `C1` base (not the report's C1/C2).

- **Round 1** (9 fresh cells around the Stage 2 winner, budget 6 each): control probed 2
  cells, 0 passes (`6/10`, `7/10`); selected probed 5, 3 passes on the new `group_size=32`
  branch (`r1-t10-v25-g32`, `r1-t15-v35-g32`, `r1-t20-v45-g32`, each `8/10`, Wilson 95%
  `[49.0, 94.3]`). Promoted `r1-t20-v45-g32` scored **83/100** (Wilson 95% `[74.5, 89.1]`)
  at **853.1 MB**, beating the `77/100` gate and saving `124.7 MB` vs `T2` (`977.8 MB`).
- **Round 2** (6 fresh cells tightening the `g32` winner, budget 4 each): both arms probed
  the 2 roots, 0 passes (`6/10`, `6/10`); no promotion. Tighter cuts fail at smoke, so the
  round stopped after 2 probes per arm.
- **Round 3** (8 fresh cells: 4 exploit ±5% text/vision MLP steps around the R1 winner,
  4 explore looser budgets, budget 8 each): both arms probed the same 3 roots then
  self-terminated (`r3-t15-v40-g32` `8/10` Wilson 95% `[49.0, 94.3]` at `885.1 MB`;
  `r3-t20-v40-g32` `7/10`; `r3-t20-v45-g64` `5/10` at `835.2 MB`). Both arms promoted the
  same cell, so one shared `100`-trial measurement covers both: **88/100** (Wilson 95%
  `[80.2, 93.0]`, `737.9 s`), beating the `77/100` gate and saving `92.7 MB` vs `T2`
  (`977.8 MB`), but `32.0 MB` larger than the R1 winner at nominally higher success
  (`88/100` vs `83/100`; separate cells, no significance claimed).
- **Round 4** (6 fresh cells on the full-vocabulary `C1` base (`int8_embedding`, no
  `compact_embedding`): 1-step solver, `group_size=32` cuts, BF16 action-attention; budget
  5 each): control and selected probed the same 5 cells, all `8/10` (Wilson 95%
  `[49.0, 94.3]`); `r4-c1-act-bf16attn-solver1` stayed unprobed. Both arms promoted the
  smallest passer, so one shared measurement covers both: `r4-c1-t15-v35-g32`
  (`C1` + `12.5%` `text_mlp` + `25%` `vision_mlp` + `group_size=32`) scored **86/100**
  (Wilson 95% `[77.9, 91.5]`, `877.3 s`) at **1199.6 MB**, beating the `77/100` gate and
  saving `88.8 MB` vs `C1` (`1288.4 MB`, `87/100`) at `-1`pp (noise). `solver_steps=1`
  verified in `deployed_architecture` (`denoising_steps` 1 vs 4) with byte-identical
  residency to `C1`; screen latency of the promoted cell is `161.6 ms` median /
  `231.9 ms` p95 (simulator excluded). Development per-task is 5/10 on task 0 and 4/10
  on task 8, the same hard tasks as R5.
- **Round 5** (6 fresh cells on untouched axes: BF16 action-attention + `solver_steps=1`,
  3-bit axis-1 HQQ vision, 12.5% attention-head pruning, 15% uniform visual tokens; budget
  5 each): control and selected probed the same 4 cells with identical scores (`r5-a15-s1`
  `7/10`; `r5-hqq3-t15-v35` and `r5-hqq3-t20-v40` `8/10`, Wilson 95% `[49.0, 94.3]`;
  `r5-heads-t15-v35` `6/10`); `r5-a20-s1` and `r5-vtok15-s1` stayed unprobed. Both arms
  promoted the same cell, so one shared measurement covers both: **85/100** (Wilson 95%
  `[76.7, 90.7]`, `920.1 s`) at **876.0 MB**, beating the `77/100` gate and saving
  `101.8 MB` vs `T2` (`977.8 MB`). Smoke failures concentrate on spatial tasks 0 and 8
  (all cells), plus 5 (action-fidelity cell) and 4 (heads cell); development per-task is
  5/10 on task 0, 7/10 on task 5, 4/10 on task 8, no better than the R3 winner (5, 9, 5).
  HQQ3 vision passes smoke and matches R1/R3 development within noise; BF16
  action-attention + solver1 (7/10) and 12.5% head pruning (6/10) do not pass; visual
  tokens remain untested live.
- **Totals:** 35 smoke runs + 4 shared development measurements on `NVIDIA GeForce RTX 3090`
  (plus 1 pre-flight smoke); reserved 500 trials untouched; phases never pooled. Replay $V$
  is not LIBERO success, and the `83/100`/`88/100` promotions do not claim a general selection gain.

Reproduce one round: `uv run --no-sync vla-rsi-round --round r1 --output studies/dream-rsi/live_round1.json`.

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
