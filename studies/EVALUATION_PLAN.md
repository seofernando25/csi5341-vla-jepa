# Evaluation plan and deliverables

Agreed scope: 2026-09-23. Owners: Noah Sprenger and Fernando Nogueira; individual assignments and course deadlines are TBD. Update checkboxes and link evidence as work completes.

Deliver a concise LaTeX paper (PDF + source) with polished vector figures. No course-specific template, page limit, or deadline has been specified; detailed logs and reproduction commands stay outside the paper.

**Question:** How much memory and inference time can we save while retaining VLA-JEPA manipulation success?

Dream-RSI is deferred. Full training from scratch is outside scope. This is a fresh evaluation campaign; previous coordination messages are not completed deliverables for it.

## Configurations

| ID | Configuration | Training |
| --- | --- | --- |
| B16 | `lerobot/VLA-JEPA-LIBERO`, BF16 | None |
| Q8 | Same checkpoint; supported Qwen linear layers quantized to 8-bit | None |
| Q4 | Same checkpoint and layer scope, 4-bit | None |
| S500 | VLA-JEPA with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and conditioning adapter, BF16 | Reduced fine-tuning |

Bitsandbytes 0.50.2 passed CUDA compatibility checks for both decoder quantization variants. Record the exact method, converted/skipped modules, compute dtype, and checkpoint revision. Keep other modules BF16. SmolVLM-backed VLA-JEPA is not SmolVLA.

## Shared controls

- Use the same GPU, software versions, LIBERO-Spatial tasks, initial states, rollout seeds, episode limits, cameras/resolution, outer observation/action preprocessing, action normalization, chunk length, and flow-matching steps. Retain and report each backbone's native image/token processor; this differs across backbone families.
- Batch size one for inference. Keep world-model loading/execution consistent across configurations; removing unused modules is a separate ablation.
- Record total/trainable parameters, checkpoint size, actual dtypes, hardware, code revision, and exact commands. Separate inference memory from training memory.
- Split demonstrations by whole trajectories; save a reproducible train/validation manifest. Use validation for checkpoint selection and reserve final simulator trials for reporting.
- Record failed runs and out-of-memory outcomes. Do not silently drop them or treat them as zero task success.

## Experiments

| ID | Question and procedure | Budget | Required evidence |
| --- | --- | --- | --- |
| E1 | Compare closed-loop manipulation success for B16/Q8/Q4/S500 on identical initial states. | Pilot: 10 tasks × 10 episodes = 100/configuration. Final target: 10 × 50 = 500/configuration, 2,000 total for four fixed checkpoints. | Per-episode success, per-task and aggregate rates, uncertainty intervals, paired differences from B16. |
| E2 | Compare inference speed and footprint on a fixed observation/instruction collection. | Warm up, then 500 action-chunk predictions × 3 repetitions/configuration. | Raw timings, median/p95 latency, peak GPU memory, checkpoint size, parameter counts. |
| E3 | Measure how reduced fine-tuning affects S500. Freeze SmolVLM initially; train its adapter and compatible action/world modules. | 500-step cost/learning pilot, then provisional checkpoints at 1,500/5,000/10,000 steps from one continuing run. | Training/validation curves, checkpoint provenance, GPU-hours, training memory, development rollout success. |

For E1, report task-macro-average success and 95% intervals; use task-stratified paired episode resampling for differences on this fixed suite. These intervals do not establish generalization to unseen tasks. Keep training-seed variation separate. A 100-episode pilot is not sufficient evidence of a small performance difference.

For E2, synchronize GPU timing. Measure model-only and full observation-to-action latency separately; exclude simulator rendering/stepping from policy latency. Time new action-chunk generation separately from buffered-action retrieval. Fix warm-up count and observation collection after the pilot; document loading time separately. Report framework peak allocated/reserved VRAM and, if available, whole-process GPU memory.

For E3, use development simulator trials for the checkpoint success curve, distinct from final reporting trials. Select the final checkpoint by validation criteria fixed before final evaluation. After measuring the pilot's runtime, freeze an affordable step/GPU-hour budget. If feasible, repeat the selected recipe with three training seeds and report each seed; these runs add training and evaluation cost beyond the table above.

The published baseline and S500 have different training histories. Interpret their comparison as deployment under limited adaptation, not a controlled causal test of backbone size alone. Do not train Qwen.

## Figures and final table

| Deliverable | Plot specification |
| --- | --- |
| F1: Success | Configuration vs success rate, with 95% intervals. |
| F2: Resources | Separate panels for median/p95 latency, peak inference VRAM, and checkpoint size. |
| F3: Main trade-off | x = inference latency; y = task success; bubble area = peak VRAM. Label configuration and hardware. |
| F4: Task breakdown | Tasks × configurations heatmap of success rates. |
| F5: Adaptation | Separate panels for S500 validation loss and development task success vs training steps. Show training-seed variation if measured. |
| T1: Summary | Configuration, precision scope, success/interval, change in percentage points, median/p95 latency, speedup, peak VRAM/reduction, checkpoint size, training budget. |

Save plots as PNG and PDF, alongside machine-readable inputs and reproducible plotting commands. Use measured values only; label missing or failed measurements explicitly.

## Delivery tracker

- [x] Agree on the three comparison arms, four configurations, and selected SmolVLM checkpoint.
- [x] Implement the SmolVLM plugin and CPU forward/backward checks. This does not establish pretrained GPU or robot performance.
- [x] Document the evaluation plan and figure list.
- [x] **D1 — Reproducible setup:** [revisions](../evaluation/sources.json), [inference/state seeds](../evaluation/protocol.json), [initial states](evaluation/initial_states.json), [trajectory split](evaluation/training_split.json), [adaptation recipe](../evaluation/training_config.json), and [budget](evaluation/budget.json) are frozen. Run records contain hardware/software versions and source hashes; [execution-source archives](evaluation/sources/index.json) preserve measured implementations.
- [x] **D2 — Working variants:** B16/Q8/Q4/S500 [CUDA checks](evaluation/checks/) passed; S500 [initialization, backward/optimizer steps and checkpoint save](evaluation/training/20260923T235515212892Z-S500-train/) verified. This establishes execution, not manipulation quality.
- [x] **D3 — Pilot:** [S500 500-step training](evaluation/training/20260923T235657557094Z-S500-train/run.json) and all four 100-episode development evaluations completed ([pilot summaries](evaluation/pilots/)); pretrained E2 timing and neural equivalence checks completed. The [final adaptation budget](evaluation/budget.json) is frozen. S500 timing at its selected checkpoint remains part of D5.
- [ ] **D4 — Adaptation:** complete E3 and select the final S500 checkpoint using the agreed validation rule. Record checkpoint location and hash; document whether training-seed repeats were feasible.
- [ ] **D5 — Final measurements:** complete E1/E2 for all four configurations under matched settings, with per-episode and per-prediction records. Document any reduced budget or unsupported configuration.
- [ ] **D6 — Analysis:** produce F1–F5 and T1 with reproduction commands and traceable inputs.
- [ ] **D7 — Submission package:** concise report covering question, method, results, limitations, and conclusion; final presentation figures; environment/configuration details and a command index. Confirm course-specific format/deadlines separately.

Recommended order: D1 → D2 → D3 → D4 → D5 → D6 → D7. Run baseline/quantization benchmarking while S500 adaptation is being prepared, using separate GPU time slots.

Current evidence: [frozen trajectory split](evaluation/training_split.json), [initial states](evaluation/initial_states.json), [measured summary](evaluation/analysis/summary.json), and [working paper](evaluation/report/report.pdf). B16/Q8/Q4 have 500 predictions × 3 repetitions in both policy-native and full pipeline timing modes. Policy-native timing includes backbone preprocessing; separate neural-forward benchmarks for all three variants also completed 500 × 3 predictions after exact agreement with native outputs on all 100 observation-bank inputs. Matched 100-episode development robot evaluations completed: [B16](evaluation/pilots/B16.json) completed 82/100 successes and [Q8](evaluation/pilots/Q8.json) completed 81/100 (paired difference −1 percentage point; 95% bootstrap interval −4 to +2). [Q4](evaluation/pilots/Q4.json) completed 81/100 (paired difference −1 point; interval −5 to +3). [Development figures F1–F4](evaluation/development/figures/) now show the three pretrained configurations. These are development diagnostics, not final scores. S500's [500-step pilot](evaluation/training/20260923T235657557094Z-S500-train/run.json) completed, with 200 held-out samples evaluated. The [frozen budget](evaluation/budget.json) retains 1,500/5,000/10,000-step checkpoints from one continuing seed-42 run (approximately ten projected GPU-hours). [F5](evaluation/adaptation/figures/F5_adaptation.pdf) includes measured validation loss through 1,500 steps and the completed step-500 development result. The [S500 step-500 checkpoint](evaluation/pilots/S500.json) completed 0/100 successful episodes (95% Wilson interval 0–3.7%), with no execution failures. This is an early adaptation diagnostic, not the selected final model. The [1,500-step continuation](evaluation/training/20260924T024324465133Z-S500-train/run.json) completed and its 100-episode development evaluation is running; training to 5,000 and 10,000 steps remains queued. Numerical figures and T1 are partial until all four configurations have matched measurements. See the [command index](evaluation/README.md).

Keep compact configs/results/figures under `studies/evaluation/` when generated. Every run needs a unique ID plus variant, checkpoint hash/revision, code revision, config, hardware, seeds, status, and artifact locations. Keep large weights, videos, and raw runtime artifacts outside Git. Do not mark a deliverable complete without linking its evidence here.

## Decision rule and optional work

Proposed engineering target: at least **25% lower peak inference memory** with at most **5 percentage points lower task success** than B16. Freeze this target before final measurements. Report estimates and uncertainty; inconclusive intervals do not prove performance preservation. The project remains a useful result if a variant fails the target and the trade-off is measured rigorously.

Only after core deliverables: quantize the trained S500 variant to test combined savings. Cross-hardware testing or unused-module removal can be separate, clearly labeled ablations. Dream-RSI is not part of this delivery plan.

References: [VLA-JEPA checkpoint](https://huggingface.co/lerobot/VLA-JEPA-LIBERO), [SmolVLM checkpoint](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [quantization backend](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Optional confirmation (2026-10-01): [n0008 cloud continuation](confirmation/n0008-20261001/cloud-results/summary.json) completed to 10k on RTX 5090. Validation loss 0.2653; ten development trials yielded zero successes. The [updated report](evaluation/report/report.pdf) includes separate loss/latency figures. Matched baseline confirmation and full n0008 development/final evaluation remain pending; this result does not complete those comparisons.

Active follow-up (2026-10-01): [SmolVLM recovery](recovery/README.md) audits action accuracy, preprocessing and gradient connectivity before further training. Recovery interventions use separate provenance and preserve the frozen RSI study. The original performance/memory decision rule remains the acceptance target.

Proposed next direction (2026-10-02): [LIBERO-initialized SmolVLM replacement](SMOLVLM_LIBERO_PROPOSAL.md). Retain compatible `VLA-JEPA-LIBERO` action/world weights, replace Qwen, and evaluate staged conditioning adaptation. Earlier initialization used `VLA-JEPA-Pretrain`; this follow-up requires separate provenance and a fixed budget before execution. No training or spending is scheduled.
