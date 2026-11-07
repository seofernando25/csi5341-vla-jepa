# Study B: post-training compression of VLA-JEPA-LIBERO

How small can the released policy get while keeping ≥90% LIBERO-Spatial success, without any
training? Completed 2026-10-10 on one RTX 3090.

Three joint recipes keep 93.4–98.0% success on 500 reserved trials at 0.94–1.38 GB peak memory
(75–83% less than B16). All of them are 1.7–1.9× slower. GB = 10^9 bytes.

## Reserved results

Same 500 official states (50 per task) for every row. Paired intervals: task-stratified bootstrap,
10,000 resamples, seed 42. Latency: 3 × 500 predictions after 20 warm-ups.

| ID | Configuration | Success [Wilson 95%] | Δ pp [paired 95%] | Gain/loss | Weights MB | Peak alloc. / reserved GB | Median / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| — | B16 control | 475/500 [92.7, 96.6] | — | — | 5532.5 | 5.605 / 5.641 | 77.5 / 80.5 |
| C1 | NF4 + channels, restricted | 467/500 [90.9, 95.3] | −1.6 [−3.2, 0.0] | 5/13 | 838.5 | 0.941 / 2.328 | 135.1 / 138.0 |
| C2 | NF4/HQQ2, restricted | 490/500 [96.4, 98.9] | +3.0 [+1.2, +5.0] | 22/7 | 977.8 | 1.066 / 1.286 | 143.5 / 146.1 |
| C3 | NF4/HQQ2, full vocabulary | 484/500 [94.9, 98.0] | +1.8 [−0.2, +3.8] | 19/10 | 1288.8 | 1.377 / 2.498 | 142.9 / 145.5 |

- **C1** removes 25% of language and 50% of vision MLP channels, quantizes language/vision blocks,
  mergers and the action head to NF4, drops the unused world model, and keeps 53 BF16 embedding
  rows. 1.61B inference parameters versus 2.28B for B16 (−29.6%).
- **C2** uses NF4 language weights, 2-bit HQQ vision blocks and action MLPs, NF4 action attention,
  interfaces and mergers. No structural removal.
- **C3** is C2 with INT8 embeddings for the full vocabulary instead of 53 restricted rows.
- "Restricted" deployments reject token IDs outside the 53 that LIBERO prompts use.
- C2's net +15 successes all come from task 5 (33→48/50); the other tasks cancel out. It is not
  evidence that 2-bit weights improve the policy.

Evidence: [per-task counts and transitions](reserved_confirmation_milestone.json),
[benchmarks](final_benchmark_milestone.json), [frozen selection](final_selection.json),
[figure](figures/confirmed_tradeoffs_v2.pdf).

## Selection pipeline

| Phase | Trials | Configurations | Purpose |
|---|---:|---:|---|
| Screen | 80 fixed frames, no simulator | 201 (198 completed, 3 runtime failures) | Memory, latency, action change vs B16 |
| Smoke | 10 | 163 | Reject gross failures |
| Development | 100 | 24 | Gate: ≥77/100 (B16 82 − 5) |
| Reserved | 500 | 4 | Final claim; frozen before outcomes |

Calibration uses the first two development observations per task (20 frames), without robot
outcomes. Development and reserved states are disjoint. Recipes, inputs, source snapshots and
tensor fingerprints were [frozen](final_selection.json) before reserved trials; reserved
outcomes guided no reselection. Per-recipe records: `measurements/<recipe>/<phase>/`; one row per
recipe in [summary.csv](summary.csv).

## Development results (100 trials)

| Configuration | Success | Weights MB | Gate |
|---|---:|---:|---|
| B16 | 82 | 5532.5 | control |
| Compact NF4/HQQ2 (→ C2) | 87 | 977.8 | pass |
| Full-vocabulary NF4/HQQ2 (→ C3) | 87 | 1288.8 | pass |
| Compact NF4 control | 85 | 1026.8 | pass |
| Compact NF4 + language/vision MLP −25/−50% (→ C1) | 82 | 838.5 | pass |
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
save 2.3% storage over C2 but drop development success 87→81. Full table with links:
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
- Pinned upstream sources: [methods.json](methods.json); kernel checks:
  [kernel_checks.json](kernel_checks.json), [lowbit_kernel_checks.json](lowbit_kernel_checks.json),
  [torchao_kernel_checks.json](torchao_kernel_checks.json).

Packed archives are rebuilt from the pinned checkpoint plus a recipe; they are not standalone
Hugging Face checkpoints. Temporary archives were deleted after exporting hashes and sizes.

## Reproduce

```bash
python -m evaluation.compression.study calibrate
python -m evaluation.compression.study run --recipe studies/compression/recipes/<id>.json --phase screen
# phases: screen, smoke, development, size, final, benchmark
python report/figures.py   # report figures from summary.csv and milestones
```

Checkpoint: `lerobot/VLA-JEPA-LIBERO` revision `735d9f692981e286ade093b5046627eda876e5d0`,
SHA-256 `b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c`. `tests/test_compression_evidence.py`
verifies [validation.json](validation.json): evidence hashes and 17 measured source files.
