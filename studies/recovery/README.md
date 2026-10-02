# SmolVLM recovery

Active goal: obtain a useful LIBERO policy while preserving the original RSI history.

**Verified bug:** the GPU processor scaled 0–1 images again, reducing vision contrast by 255×. The plugin is fixed and a real-processor regression test passes. Old GPU-study results remain recorded but are confounded; they do not establish that SmolVLM is inadequate.

Corrected inputs plus four trainable decoder layers passed the sixteen-frame overfit gate after 1,250 updates: 94% lower arm error and 0.9% gripper errors. This is training-only evidence, not successful control. A registered recovery run on the full split and development rollouts are next.

- [Methodology, evidence and remaining questions](METHODOLOGY.md)
- [Compact diagnostics and provenance](diagnostics/)
- [Action-accuracy vector figure](figures/F7_action_audit.pdf), rebuilt with `python -m evaluation.recovery_analysis`

Final target: at least 25% less inference memory and no more than five points below matched B16 success. Vast is off; USD2.17 spent, USD11.83 remains authorized.

The [recovery recipe](../../evaluation/recovery_config.json) uses batch eight, four trainable decoder layers and fresh AdamW (decoder 1e-5; adapter/action/world 1e-4). Its fixed 20k-update schedule resumes at 2k/5k/10k milestones. Held-out physical action error selects the checkpoint; final LIBERO remains reserved. Native batch-eight training, exact weight restoration, optimizer resume and 200-frame validation passed locally.
