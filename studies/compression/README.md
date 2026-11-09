# Study B: post-training compression of VLA-JEPA-LIBERO

How small can the released policy get while keeping ≥90% LIBERO-Spatial success, without any
training? Completed 2026-10-10 on one RTX 3090.

Two joint recipes (C1, C2) keep 96.8–98.0% success on 500 reserved trials at 1.38–1.51 GB peak
memory (73–75% less than B16) and use no calibration data. C1 is 1.8× slower than B16 on the stock
path; C2 runs at 31.6 ms through a CUDA-graph engine. GB = 10^9 bytes.

The report renumbers the joint arms: its C1 is the full-vocabulary NF4/HQQ2 recipe
(`deploy-hqq2-vision-action`, called C3 here before the renumbering) and its C2 is the data-free
graph-engine recipe (`fast-nf4-all-solver1-packed`, formerly C4). The restricted variants below
keep their evidence as T1/T2 and are not reported.

## Reserved results

Same 500 official states (50 per task) for every row. Paired intervals: task-stratified bootstrap,
10,000 resamples, seed 42. Latency: 3 × 500 predictions after 20 warm-ups.

| ID | Configuration | Success [Wilson 95%] | Δ pp [paired 95%] | Gain/loss | Weights MB | Peak alloc. / reserved GB | Median / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| — | B16 control | 475/500 [92.7, 96.6] | — | — | 5532.5 | 5.605 / 5.641 | 77.5 / 80.5 |
| C1 | NF4/HQQ2, full vocabulary | 484/500 [94.9, 98.0] | +1.8 [−0.2, +3.8] | 19/10 | 1288.8 | 1.377 / 2.498 | 142.9 / 145.5 |

- **C1** uses NF4 language weights, 2-bit HQQ vision blocks and action MLPs, NF4 action attention,
  interfaces and mergers, and INT8 embeddings for the full vocabulary. No structural removal, no
  calibration data; it drops the unused world model.
- C1's net +9 successes split into +11 in task 5 (33→44/50) and −2 across the other nine tasks. It
  is not evidence that quantization improves the policy.

Evidence: [per-task counts and transitions](reserved_confirmation_milestone.json),
[benchmarks](final_benchmark_milestone.json), [frozen selection](final_selection.json).
Figures: [final report](../../report/main.pdf).

## Graph execution (C2)

The slowdown above comes from launch overhead, not arithmetic: the stock path launches about 5,000
kernels per prediction and synchronizes with the host inside the vision tower and decoder
([launch profile](../evaluation/launch_profile.json)). `evaluation/fastpath.py` captures one CUDA
graph per instruction, fuses elementwise kernels with `torch.compile`, and dequantizes NF4 with one
Triton kernel (`evaluation/nf4.py`) that is bit-identical to bitsandbytes. The first call per
instruction captures and compiles (about 11 s each; priming ten prompts took 109 s), which the
latencies exclude.

**C2** (`fast-nf4-all-solver1-packed`) is data-free: NF4 on every module, INT8 embedding table (full
vocabulary), no pruning, no calibration data, one flow step instead of four, packed weights.

| Arm | Success /500 | Δ vs B16 pp [paired 95%] | Gain/loss | Median / p95 ms | Peak GB (alloc. / reserved) |
|---|---:|---:|---:|---:|---:|
| B16, stock | 475 | — | — | 77.5 / 80.5 | 5.61 / — |
| B16, graph engine (`fast-b16-graph`) | 474 | −0.2 [−1.2, +0.8] | 3/4 | 28.5 / 28.9 | 5.70 / — |
| NF4 all, 4 steps (`fast-nf4-all-packed`) | 486 | +2.2 [+0.8, +3.8] | 14/3 | 36.8 / 37.7 | 1.51 / — |
| **C2**, NF4 all, 1 step | 490 | +3.0 [+1.4, +4.8] | 18/3 | 31.6 / 32.0 | 1.51 / 2.71 |

Timing-only controls (benchmark phase): B16 graph with one step 24.8 ms; NF4 all, one step, with
dense bf16 copies of the weights (`fast-nf4-all-solver1`) 24.7 ms at 5.45 GB. Development scores:
C2 88/100, four-step NF4 83, engine-B16 81 (stock B16: 82).

The engine does not change success (engine-B16 vs stock B16). The +2.2 to +3.0 points appear with
quantized weights and are concentrated in task 5 (C2: 11 of a net 15 successes), as they were for
C1. We report this as unexplained; it is not evidence that quantization improves the policy.

**Task-specific variants (T1/T2, not used in the report).** T1
(`deploy-nf4-vision-text-channels-restricted`) is the smallest arm: 467/500 (93.4%, −1.6
[−3.2, 0.0]), 0.941 GB peak, 135.1 ms. It removes 25% of language and 50% of vision MLP channels,
quantizes language/vision blocks, mergers and the action head to NF4, drops the unused world model,
and keeps 53 BF16 embedding rows (1.61B inference parameters versus 2.28B, −29.6%). T2
(`deploy-hqq2-vision-action-restricted`) is C1's recipe with the same restricted vocabulary:
490/500 (98.0%, +3.0 [+1.2, +5.0]), 1.066 GB, 143.5 ms. Both reject token IDs outside the 53 that
LIBERO prompts use, and T1's channel pruning is calibrated on 20 development observations of the
same tasks. `fast-nf4-solver1` (dense weights: 477/500, 20.0 ms, 4.28 GB) and
`fast-nf4-solver1-packed` (476/500, 28.8 ms, 1.02 GB) run T1's recipe with one flow step. They are
kept as evidence; the 1.02 GB figure depends on the restricted vocabulary.

Reproduce: `python -m evaluation.compression.study run --recipe studies/compression/recipes/<id>.json --phase <phase>`.

## Selection pipeline

| Phase | Trials | Configurations | Purpose |
|---|---:|---:|---|
| Screen | 80 fixed frames, no simulator | 201 (198 completed, 3 runtime failures) | Memory, latency, action change vs B16 |
| Smoke | 10 | 163 | Reject gross failures |
| Development | 100 | 24 | Gate: ≥77/100 (B16 82 − 5) |
| Reserved | 500 | 4 | Final claim; frozen before outcomes |

Calibration uses the first two development observations per task (20 frames), without robot
outcomes. Development and reserved states are disjoint. The selected recipes were [frozen](final_selection.json) before reserved trials; reserved
outcomes guided no reselection. Per-recipe records: `measurements/<recipe>/<phase>/`; one row per
recipe in [summary.csv](summary.csv).

## Development results (100 trials)

| Configuration | Success | Weights MB | Gate |
|---|---:|---:|---|
| B16 | 82 | 5532.5 | control |
| Compact NF4/HQQ2 (→ T2) | 87 | 977.8 | pass |
| Full-vocabulary NF4/HQQ2 (→ C1) | 87 | 1288.8 | pass |
| Compact NF4 control | 85 | 1026.8 | pass |
| Compact NF4 + language/vision MLP −25/−50% (→ T1) | 82 | 838.5 | pass |
| Compact NF4 + mild structure | 81 | 900.8 | pass |
| Compact HQQ3/HQQ2 | 81 | 955.1 | pass |
| Compact NF4 + language MLP −25% | 78 | 890.5 | pass |
| Language NF4 only | 81 | 2474.4 | pass |
| Vision MLP channels −50% | 83 | 5331.1 | pass |
| Language KV heads −25% | 83 | 5356.3 | pass |
| Vision heads −50% | 82 | 5431.8 | pass |
| Action MLP channels −75% | 80 | 5419.2 | pass |
| Language layers −4/28 (CKA ranking) | 80 | 5129.8 | pass |
| Vision layers −12/24 | 80 | 5230.2 | pass |
| Action layers −8/16 | 77–80 | 5392.4 | pass |
| Merger channels −75% | 76 | 5381.5 | fail |
| Compact NF4 + combined channels | 70 | 770.2 | fail |
| Action heads −75% | 69 | 5452.2 | fail |
| Compact NF4 + combined depth | 65 | 808.2 | fail |

Individually safe cuts fail when combined; the failures bound the tested recipes, not a global
minimum. Lower nominal bit width does not guarantee a better deployment: HQQ3 language weights
save 2.3% storage over T2 but drop development success 87→81. Full table with links:
[development milestones](development_refinement_outcomes.json), [decisions](decisions.jsonl).

## Coverage

All 21 component domains have closed-loop evidence ([coverage.json](coverage.json)): language,
vision and action MLP/attention/heads/depth, mergers, embeddings, conditioning and visual tokens,
action interfaces, solver steps, action and replanning horizons, and world-model residency.
Solver, horizon and token reductions have only 10-trial evidence ([solver](solver_milestone.json),
[execution](execution_milestone.json), [tokens](token_milestone.json)).

## Methods and their limits

- bitsandbytes LLM.int8/NF4; HQQ 2–4 bit with native and fused GemLite kernels; TorchAO W8A8.
- GPTQ and SparseGPT from pinned upstream code; Wanda and magnitude masks. Dense masks save no
  memory and carry no storage or speed claim; packed 2:4 sparsity uses the sparse CUDA backend.
- Adaptations, not reproductions: AWQ diagonal scaling (not the block-output search); QuantVLA
  ATM/OHB with NF4/BF16 (not its W4A8 pipeline); CKA layer ranking without the paper's fine-tuning.
- Upstream sources are pinned by repository and revision in [methods.json](methods.json) and
  cloned on first use (`evaluation/compression/vendor.py`). Kernel checks:
  [kernel_checks.json](kernel_checks.json), [lowbit_kernel_checks.json](lowbit_kernel_checks.json),
  [torchao_kernel_checks.json](torchao_kernel_checks.json).

Packed archives are rebuilt from the pinned checkpoint plus a recipe; they are not standalone
Hugging Face checkpoints. Temporary archives were deleted after exporting hashes and sizes.

## Reproduce

```bash
python -m evaluation.run collect                        # observation bank (Study A)
python -m evaluation.compression.study calibrate
python -m evaluation.compression.study run --recipe studies/compression/recipes/<id>.json --phase screen
# phases: screen, smoke, development, size, final, benchmark
python -m evaluation.compression.analysis               # rebuild summary.csv
```

Checkpoint: `lerobot/VLA-JEPA-LIBERO` revision `735d9f692981e286ade093b5046627eda876e5d0`.
The campaign scheduler that chose which recipes advanced was removed after the study closed;
its decisions are recorded in [decisions.jsonl](decisions.jsonl) and the `*_selection.json` files,
and the code is in Git history at commit `82df05d`.
