# VLA-JEPA-LIBERO: Maximum Compression at ≥90% Success

**Completed October 10, 2026.** The smallest confirmed configuration uses **838.5 MB packed weights**, reaches **467/500 (93.4%) LIBERO-Spatial success**, and reduces warm peak CUDA allocation **83.2%** relative to the fresh B16 control. Its median pipeline latency increases from **77.5 to 135.1 ms**. Compact NF4/HQQ2 achieves the highest observed success, **490/500 (98.0%)**, and the lowest allocator-reserved peak, **1.286 GB**.

The study covers **201 terminal screens (198 completed, three runtime failures), 163 ten-trial configurations, 24 matched 100-trial development configurations, four 500-trial reserved confirmations and four 3,000-prediction benchmarks**. All **21 major component domains** have closed-loop task evidence. The [final data audit](final_benchmark_milestone.json) verifies 6,030 episode records across nested smoke/development and disjoint reserved cohorts, 12,000 timing rows, 534 frozen scientific sources, seven frozen inputs and matching deployed tensors. Runtime failures remain separate from architecture outcomes.

## Reserved results and measured inference

All configurations start from the released LIBERO-trained Qwen3-VL-2B policy; no optimizer training or fine-tuning is used. Each reserved result uses the same 500 official states, 50 per task. Measurements below are actual final benchmarks on one RTX 3090. MB/GB are decimal; CUDA allocated and allocator-reserved peaks are distinct from whole-process NVML memory.

| Configuration | Reserved success | Packed weights, MB | Peak allocated / reserved, GB | Pipeline median / p95, ms |
|---|---:|---:|---:|---:|
| B16 control | 475/500 (95.0%) | 5532.5 | 5.605 / 5.641 | 77.5 / 80.5 |
| NF4 + channels, restricted | 467/500 (93.4%) | 838.5 | 0.941 / 2.328 | 135.1 / 138.0 |
| NF4/HQQ2, restricted | 490/500 (98.0%) | 977.8 | 1.066 / 1.286 | 143.5 / 146.1 |
| NF4/HQQ2, full vocabulary | 484/500 (96.8%) | 1288.8 | 1.377 / 2.498 | 142.9 / 145.5 |

| Compressed configuration | Success Wilson 95% interval | Paired change vs B16, pp | Paired 95% interval, pp |
|---|---:|---:|---:|
| NF4 + channels, restricted | 90.9–95.3% | -1.6 | -3.2 to +0.0 |
| NF4/HQQ2, restricted | 96.4–98.9% | +3.0 | +1.2 to +5.0 |
| NF4/HQQ2, full vocabulary | 94.9–98.0% | +1.8 | -0.2 to +3.8 |

The smallest deployment removes 25% of language MLP channels and 50% of vision MLP channels, quantizes language/vision blocks, mergers and the action head with NF4, removes unused inference world-model residency, and retains 53 BF16 embedding rows. It has **84.84% smaller packed weights**, **83.22% lower allocated peak**, **58.74% lower reserved peak**, and **74.34% higher median pipeline latency**. Compact NF4/HQQ2 instead uses NF4 language weights, HQQ2 vision blocks/action MLPs and NF4 action attention/interfaces/mergers, without structural channel removal. Its weights are 82.33% smaller and allocated peak is 80.98% lower. All three selected compressed policies clear the 450/500 success target, remain within the earlier five-percentage-point success-loss limit, and exceed the 25% allocated-memory reduction target.

The smallest recipe represents 1,606,704,391 inference-core parameter elements versus 2,282,714,375 for B16 (29.61% fewer). This combines physical MLP channel removal with vocabulary compaction. A further 483,423,616 world-model parameter elements leave inference residency. NF4 packing changes storage precision; world-model removal changes residency. Their savings are attributed separately in the recipes and parameter census.

Both restricted deployments reject unseen token IDs. The full-vocabulary arm replaces those 53 rows with INT8 embeddings covering the original vocabulary; coverage and embedding precision change together. It scores 96.8%, with 76.71% smaller weights and 75.44% lower allocated peak. It recovered three compact-arm failures and lost nine compact-arm successes (−1.2 points, paired interval −2.6 to 0). Full vocabulary preserves input coverage; success on new tasks or other LIBERO suites was not tested.

[Final tradeoff figure](figures/confirmed_tradeoffs_v2.pdf) · [figure provenance](final_figure_evidence_v2.json) · [all per-task counts and paired transitions](reserved_confirmation_milestone.json). Paired intervals use 10,000 task-stratified resamples, seed 42. Policy-only medians are 76.3, 134.0, 142.3 and 141.7 ms in table order. Each mode uses 500 predictions × three repetitions, with 20 warmups per repetition. All 100 observation-bank shapes are primed first; priming time and memory are recorded separately. Loading/PTQ initialization and simulator throughput are outside warm inference latency.

## Development evidence and compression limits

The 100-trial B16 control scores 82/100; the preregistered relative gate is 77/100. These scores are selection evidence, separate from the 500-trial target. Standalone structural candidates retain BF16 weights and world-model residency. Compact rows use restricted vocabulary and remove world-model residency. Screen peaks below are preliminary CUDA allocated memory, in GiB; final benchmark peaks appear above.

| Configuration | Success /100 | Packed archive, MB | Screen peak, GiB | Gate |
|---|---:|---:|---:|---|
| [B16](baseline_development_milestone.json) | 82 | 5532.5 | 5.220 | control |
| [Compact NF4 + language/vision MLP −25%/−50%](development_refinement_outcomes.json) | 82 | 838.5 | 0.876 | pass |
| [Compact NF4 + language MLP −25%](text_channel_development_milestone.json) | 78 | 890.5 | 0.918 | pass |
| [Compact NF4 + mild structure](mild_structural_development_milestone.json) | 81 | 900.8 | 0.984 | pass |
| [Compact NF4/HQQ2](hqq2_development_milestone.json) | 87 | 977.8 | 0.993 | pass |
| [Compact HQQ3/HQQ2](hqq3_development_milestone.json) | 81 | 955.1 | 1.035 | pass |
| [Compact NF4 control](nf4_development_milestone.json) | 85 | 1026.8 | 1.052 | pass |
| [Full-vocabulary NF4 + mild structure](full_vocabulary_structural_development_milestone.json) | 82 | 1211.7 | 1.275 | pass |
| [Language NF4; other roots BF16](conservative_nf4_development_milestone.json) | 81 | 2474.4 | 2.392 | pass |
| [Full-vocabulary NF4/HQQ2](full_vocabulary_hqq2_development_milestone.json) | 87 | 1288.8 | 1.282 | pass |
| [Action MLP channels −75%](action_mlp_development_milestone.json) | 80 | 5419.2 | 5.116 | pass |
| [Merger channels −75%](merger_development_milestone.json) | 76 | 5381.5 | 5.079 | fail |
| [Language MLP channels −25%](text_mlp_development_milestone.json) | 79 | 5004.0 | 4.726 | pass |
| [Vision MLP channels −50%](vision_mlp_development_milestone.json) | 83 | 5331.1 | 5.033 | pass |
| [Action heads −75%](action_heads_development_milestone.json) | 69 | 5452.2 | 5.140 | fail |
| [Language KV heads −25%](text_heads_development_milestone.json) | 83 | 5356.3 | 5.068 | pass |
| [Vision heads −50%](vision_heads_development_milestone.json) | 82 | 5431.8 | 5.140 | pass |
| [Action layers −8/16, residual ranking](action_layers_development_milestone.json) | 77 | 5392.4 | 5.087 | pass |
| [Language layers −4/28, CKA ranking](text_layers_development_milestone.json) | 80 | 5129.8 | 4.841 | pass |
| [Vision layers −12/24, residual ranking](vision_layers_development_milestone.json) | 80 | 5230.2 | 4.938 | pass |
| [Action layers −8/16, CKA ranking](action_cka_layers_development_milestone.json) | 80 | 5392.4 | 5.087 | pass |
| [Full-vocabulary HQQ3/HQQ2](full_vocabulary_hqq3_development_milestone.json) | 82 | 1266.1 | 1.306 | pass |
| [Compact NF4 + combined channels](joint_channel_development_milestone.json) | 70 | 770.2 | 0.810 | fail |
| [Compact NF4 + combined depth](joint_depth_development_milestone.json) | 65 | 808.2 | 0.842 | fail |


All 24 development/size pairs have matching tensor fingerprints. The [selection review](pre_confirmation_review.json), [initial audit](initial_development_size_milestone.json), [supplemental audit](supplemental_development_size_milestone.json), [refinement outcome](development_refinement_outcomes.json), [development figure](figures/development_tradeoffs_v2.pdf) and [decisions](decisions.jsonl) preserve the measured selection history.

Removing 25% of language MLP channels, 50% of vision MLP channels, 25% of language KV heads, 50% of vision heads, four language layers, twelve vision layers or eight action layers clears the development gate when tested individually. Removing 75% of merger channels or action heads fails that gate. Their combinations are measured independently: the 770.2 MB combined-channel and 808.2 MB combined-depth deployments score 70/100 and 65/100. These failures bound the tested recipes; they do not establish a global minimum. The 838.5 MB recipe is the smallest registered configuration confirmed on reserved trials.

The final refinement adds only vision MLP pruning to identical retained language channels: 78→82 successes, four recovered failures and no lost successes (+4 points; paired interval +1 to +8), with 5.85% lower resident storage. More aggressive head combinations score 2/10 and 0/10 in the fixed refinement smoke round and do not advance. HQQ3 language substitution saves only 2.3% resident storage relative to compact HQQ2, while development success falls 87→81 and preliminary latency rises 33.5%. Lower nominal bit width does not guarantee a better measured deployment.

## Measured component coverage

Counts overlap because a joint deployment changes several components. “Rollout” includes at least ten matched task trials; development and confirmation require 100 and 500. A component can have task evidence without individual reserved confirmation. The machine-readable [coverage ledger](coverage.json) links every recipe and phase record.

| Component domain | Completed screens | Rollout configurations | Development /100 | Confirmed /500 |
|---|---:|---:|---:|---:|
| language MLP | 55 | 51 | 13 | 3 |
| language attention | 51 | 46 | 12 | 3 |
| language heads | 6 | 6 | 1 | 0 |
| language depth | 14 | 11 | 4 | 0 |
| vision MLP | 49 | 44 | 12 | 3 |
| vision attention | 45 | 41 | 11 | 3 |
| vision heads | 7 | 6 | 1 | 0 |
| vision depth | 14 | 11 | 4 | 0 |
| main/DeepStack mergers | 29 | 27 | 12 | 3 |
| embeddings | 24 | 24 | 11 | 3 |
| conditioning tokens | 4 | 4 | 0 | 0 |
| visual tokens | 6 | 4 | 0 | 0 |
| action MLP | 54 | 48 | 12 | 3 |
| action attention | 36 | 33 | 11 | 3 |
| action heads | 5 | 3 | 1 | 0 |
| action depth | 13 | 10 | 5 | 0 |
| action interfaces | 36 | 33 | 11 | 3 |
| solver steps | 3 | 3 | 0 | 0 |
| generated action horizon | 3 | 3 | 0 | 0 |
| replanning horizon | 3 | 3 | 0 | 0 |
| world-model residency | 26 | 26 | 12 | 3 |

Solver, action-horizon and token reductions have ten-trial task evidence rather than reserved confirmation. One/two/three solver steps each score 8/10; one-action generation scores 8/10, while one-action replanning scores 6/10. Conditioning reductions to 24/16/eight tokens score 8/10, 7/10 and 0/10. Latent-prompt removal scores 8/10. Visual-token removal of 50% scores 8/10 for either tested ranking; 75% scores 6/10 by magnitude and 7/10 uniformly. Their [solver](solver_milestone.json), [execution](execution_milestone.json) and [token](token_milestone.json) records are preliminary evidence. Conditioning changes prompt occurrences; visual tokens are removed after full vision encoding, preserving positions and DeepStack membership. These transforms do not prune learned weights or the vision encoder's compute.

## Methods, attribution and protocol

- [Lottery ticket hypothesis](https://arxiv.org/abs/1803.03635): concerns sparse subnetworks and their original initialization/retraining. This study measures compression of trained weights; it does not test initialization rewinding.
- [LLM.int8](https://arxiv.org/abs/2208.07339), [NF4](https://arxiv.org/abs/2305.14314), [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323): weight/activation and calibrated weight quantization. NF4 deployment uses bitsandbytes; method adaptations and kernel support are recorded explicitly.
- [Wanda](https://arxiv.org/abs/2306.11695), [SparseGPT](https://arxiv.org/abs/2301.00774): activation-aware and Hessian-compensated weight sparsification, including 2:4 patterns. Structural channel/head/layer reductions are tracked separately.
- [HQQ](https://github.com/dropbox/hqq), [GemLite](https://github.com/dropbox/gemlite): gradient-free proximal quantization with physically packed 2/3/4-bit weights. Native HQQ and fused GemLite 2/4-bit CUDA deployments are separate tests; metadata and unsupported layer widths are included in the accounting.
- [EcoVLA](https://arxiv.org/abs/2602.00780), [CKA layer pruning](https://arxiv.org/abs/2606.20246): current VLA channel/depth methods. The latter's reported downstream fine-tuning is distinct from our post-training tests.
- [QuantVLA](https://arxiv.org/abs/2602.20309), [CHASE-VLA](https://arxiv.org/abs/2610.02666): current action-expert PTQ methods. Track exact implementations, adaptations, and availability; generic low-bit quantization is not labeled as a full reproduction of these methods.
- [QUOTA](https://arxiv.org/abs/2604.17320): joint visual-token pruning and quantization. Its paper says code will be released; verify availability and Qwen/VLA compatibility before claiming reproduction.


GPTQ and SparseGPT use pinned upstream algorithms with sequential calibration on 20 frozen development observations. Packed INT4 uses the Ampere-compatible ATen kernel; packed 2:4 uses the sparse CUDA backend. Dense magnitude/Wanda masks carry no storage or speed claim. Native HQQ, GemLite and TorchAO eager/scoped compiled deployments have separate recorded kernel and task checks. Scoped Linear compilation is not whole-policy fusion; the failed default action W8A8 route is preserved alongside the successful Triton route. See [methods](methods.json), [kernel checks](kernel_checks.json), [TorchAO checks](torchao_kernel_checks.json), [HQQ/GemLite checks](lowbit_kernel_checks.json) and [compiled deployment outcomes](compiled_deployment_milestone.json).

AWQ diagonal reconstruction is an adaptation, not the original block-output search. QuantVLA ATM/OHB uses its pinned Diffusers processor and published ratio/clamp rules with NF4/BF16; it does not reproduce the paper's complete GR00T DuQuant/W4A8 pipeline. CKA pruning omits the paper's downstream fine-tuning. The recorded primary-source review found EcoVLA's implementation release pending and QUOTA's code promised; CHASE-VLA is reviewed but unimplemented. These are explicit method-coverage limits, not claimed reproductions. This is post-training compression of learned weights, not an initialization-rewinding test of the lottery-ticket hypothesis.

The original policy is `lerobot/VLA-JEPA-LIBERO`, revision `735d9f692981e286ade093b5046627eda876e5d0`, checkpoint SHA256 `b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c`. LeRobot stays pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`. The trained policy restores with no missing learned keys; the identical tied-embedding alias and unused checkpoint-only world-predictor state/extrinsics entries are documented. The accepted project protocol casts parameters and buffers, including rotary buffers, to BF16; retains seven generated/executed actions, four Euler steps and the 280-step project rollout cap. These are measurements of the pinned LeRobot port and local evaluator, not a claim of exact author-evaluator reproduction.

Calibration uses the first two development observations per task (20 total), without robot outcomes; diagnostics use the other 80. Development and official reserved state cohorts are disjoint. Native reset-state hashes and trial seeds are verified. Recipes, seven inputs, source snapshots and actual tensor fingerprints were [frozen before reserved outcomes](final_selection.json). Final results guide no tuning or reselection. Actual packed archive sizes include safetensors headers and packed quantization state; subclass/alias layouts are recorded in separate sidecars. These archives require reconstruction from the pinned checkpoint plus recipe and are not directly loadable Hugging Face checkpoints. Compact hash/size/layout evidence was exported before deleting temporary weight archives.

Original B16/Q8/Q4, SmolVLM and Dream-RSI studies remain separate. Qwen is evaluation-only; the active SmolVLM plugin and RSI protocol are untouched. No commits, pushes or board publication were made. [Validation](validation.json) records 59 passing tests and lint; all 17 validated compression source files still match. [Completion audit](completion_audit.json) records requirement-level evidence. All controllers are terminal; no compression GPU job remains active.
