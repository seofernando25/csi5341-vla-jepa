# n0008 confirmation

Train the selected late-decoder residual adapter and a matched GPU SmolVLM baseline from the same pretrained action/world initialization. The frozen [plan](plan.json) uses seed 42, batch 2, a fixed 10k-step schedule, the existing trajectory split and 200 held-out observations. Select the lowest held-out loss at 1,500/5,000/10,000 steps; simulator success never selects weights.

The runner resumes native optimizer checkpoints after interruption, then evaluates each selected model on the same 100 development LIBERO-Spatial episodes and 500 × 3 timing inputs. Preserve separate labels and source/checkpoint hashes. Final 500-episode evaluation follows development review. Qwen remains evaluation-only; the published 95% project reference has a different training history.

This confirmation uses the established longer adaptation recipe, not the short RSI screening schedule. Screening ranks are hypotheses; compare these two confirmation arms directly. The initial budget is much smaller than the paper's training exposure. No task-performance claim exists until rollouts complete.

Runtime weights/logs live under ignored `outputs/`; compact selections and measurements remain here and in `studies/evaluation/`. Pruning preserves milestone inference weights and the latest optimizer state after exporting hashes.

Cloud continuation completed on RTX 5090: 10k held-out loss 0.2653; development 0/10 successes; pipeline median/p95 302.8/407.7 ms; inference allocation 2.97 GiB. The full resumable checkpoint was verified locally before rental deletion (total spend about $2.17). These are separate diagnostics, not final LIBERO or a matched hardware/pipeline comparison. See [compact evidence and figure](cloud-results/summary.json).
