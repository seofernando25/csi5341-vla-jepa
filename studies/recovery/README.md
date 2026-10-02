# SmolVLM recovery

Active goal: obtain a useful LIBERO policy while preserving the original RSI history.

**Verified input bug:** the GPU processor scaled 0–1 images again, reducing vision contrast by 255×. The plugin is fixed and a real-processor regression test passes. Old GPU-study results remain recorded but are confounded; they do not establish that SmolVLM is inadequate.

Corrected inputs plus four trainable decoder layers passed the sixteen-frame overfit gate after 1,250 updates: 94% lower arm error and 0.9% gripper errors. This is training-only evidence. The separate full-split recovery is running; development rollouts test actual control at registered milestones.

- [Methodology, evidence and remaining questions](METHODOLOGY.md)
- [Compact diagnostics and provenance](diagnostics/)
- [Action-accuracy vector figure](figures/F7_action_audit.pdf), rebuilt with `python -m evaluation.recovery_analysis`
- [Recovery curves](figures/F8_recovery_progress.pdf), rebuilt from complete evaluations with `python -m evaluation.recovery_progress`

Final target: at least 25% less inference memory and no more than five points below matched B16 success. Recovery training is running on one RTX 5090 at USD0.526/hour. Total spending remains capped at USD14; native state is exported before deletion, with provider cleanup October 2 at 20:00 UTC.

The [recovery recipe](../../evaluation/recovery_config.json) uses batch eight, four trainable decoder layers and fresh AdamW (decoder 1e-5; adapter/action/world 1e-4). Its fixed 20k-update schedule resumes at 2k/5k/10k milestones. Held-out physical action error selects the checkpoint; final LIBERO remains reserved. Native batch-eight training, exact weight restoration, optimizer resume and 200-frame validation passed locally.

For failure diagnosis, `python -m evaluation.control_trace --help` records one development episode's commands, robot states and sparse camera frames through the unchanged evaluator. Raw traces stay in ignored runtime storage; they are not timing benchmarks or final trials.

Recovery achieved **1/10 development successes at 2k, 0/10 at 5k and 1/10 at the 10k milestone**, despite improving held-out action error. The [paired task-1 trace](diagnostics/task1-control-case/) shows Qwen completing in 113 steps while RGB-2k fails at 280, with 1 versus 28 gripper-command transitions. This identifies a control symptom in one case; it does not establish the sole failure cause. Training continues to the registered 20k endpoint.

**Inference precision mismatch:** the legacy loader rounds nonpersistent rotary frequencies to BF16, unlike native training. A reversible fixed-frame audit confirms higher action loss. A [separate paired development test](native_rope_registration.json) preserves original FP32 frequencies; the same fixed 5k checkpoint achieves 1/10 development successes in both modes on the local RTX3090. The defect is real but has not rescued control. Historical diagnostics retain their original loader provenance.
