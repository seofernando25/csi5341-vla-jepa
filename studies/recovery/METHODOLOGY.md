# Recovery methodology and evidence

Active goal, requested 2026-10-01: diagnose failed control and develop an evaluated SmolVLM policy. The original RSI journal, candidates, checkpoints and results remain intact. New interventions are separate recovery experiments.

**Acceptance:** retain the project's target of at least 25% lower peak inference memory with no more than a five-point success loss against B16, measured on matched hardware and final LIBERO trials. Lower training loss alone does not pass. B16's recorded final result is 95%; recovery targets at least 90%, with uncertainty reported.

## Evidence so far

| Check | Finding |
| --- | --- |
| Cloud lifecycle | Rental deleted after twelve exported checkpoint/optimizer files were hash-verified. Total spending USD2.17; USD11.83 remains authorized. |
| Demonstration actions | On twenty selected held-out frames, n0008-10k arm-command MSE is 0.04796, versus B16 0.000556 and a training-set mean predictor 0.07994. These are simulator command units, not metres. |
| Action processing | Expert labels round-trip with maximum arm error below 1e-7 and no valid-frame gripper errors. The pinned normalizer ignores the saved gripper mask, producing -1/+1 labels rather than the original 0/1 convention. This complicates loss comparison, but does not break the tested command round-trip. |
| Camera/state conventions | Dataset images visually match the simulator after the existing 180-degree rotation. The existing environment processor applies that rotation and constructs the same eight state fields. No verified convention error yet. |
| Gradient flow | One native action-loss backward pass gives finite, nonzero gradients to both adapters and the action head. Special-token embeddings remain frozen. |
| Conditioning | Altering images, instruction or state changes actions. This sensitivity does not establish useful visual grounding; black-image and zero-state probes are outside the training distribution. |
| **Image scaling bug** | The GPU call mixes flat `do_rescale=False` with structured `images_kwargs.device`. Transformers ignores the flat image argument when structured image arguments exist. A real training frame reaches vision with range -1.0005 to -0.9923 (standard deviation 0.00153), destroying most contrast. Moving the flag into `images_kwargs` fixes the real-processor black/white regression to -1/+1. |
| Frozen-backbone overfit | After 250 native updates on sixteen training frames, arm MSE fell only from 0.03467 to 0.02478 and gripper error rose from 6.25% to 13.39%. The declared overfit gate failed. This limited diagnostic does not establish impossibility. |
| Partial-backbone overfit | Unfreezing four decoder layers under the legacy image pipeline also failed the same gate (arm MSE 0.03112; gripper error 7.14%). This cannot evaluate backbone adaptation with correctly processed images. |
| Corrected-pipeline overfit | With frozen decoder layers, 250 updates gave arm MSE 0.02325 and 7.14% gripper errors. With four trainable layers, the extended probe stopped early at 1,250 updates: arm MSE 0.002628 versus initial 0.04225 (94% lower), with 0.89% gripper errors. The declared overfit gate passed. This is training-only evidence, not successful control. |
| Native image geometry | Two 224-pixel camera images expand to 34 crops at 512 pixels and 2,266 tokens in the geometry-only probe. Disabling native splitting gives two crops and 146 tokens; these counts omit the longer policy-specific prompt. This suggests an efficiency ablation, not a measured speedup. It would require a separately registered processing variant and new training/evaluation, outside frozen RSI candidates. |

The [compact diagnostics](diagnostics/) include checkpoint/source hashes, frame membership and measurements. Qwen's training membership is unknown; this is an action-accuracy diagnostic, not a generalization comparison. Normalized all-dimension MSE must not rank the models because their gripper-label conventions differ. The verified image bug is a major confound for GPU-study architecture claims; old outcomes do not establish a limitation of SmolVLM itself.

The overfit probes use fixed seeds but do not enforce bitwise deterministic CUDA training. They are single-run engineering diagnostics, not replicated causal estimates of an architecture's benefit.

The [paper](https://arxiv.org/html/2602.10098v1#A2) reports 30k simulation fine-tuning updates at global batch 256, with trainable VLM parameters. Our 10k updates at batch two expose 384 times fewer frame samples and freeze SmolVLM. It also substitutes generic SmolVLM representations for robotics-adapted Qwen representations. Insufficient adaptation is plausible; model size alone has not been established as the cause. The authors' current [training configuration](https://github.com/ginwind/VLA-JEPA/blob/main/scripts/configs/vlajepa_libero_ft.yaml) differs from the appendix; report each recipe explicitly.

[SmolVLA's primary paper](https://arxiv.org/html/2506.01844v1#S4.SS3) reports 90% LIBERO-Spatial success for its 450M model, with frozen VLM and 100k simulation updates at batch 64. Its conditioning uses earlier visual/language features, no image tiling, a different action expert and more frequent feedback. This is evidence that small SmolVLM-based control can work, not evidence that our transplanted VLA-JEPA architecture should match it. The registered 20k×8 recovery still exposes 40× fewer frame samples than that recipe and 48× fewer than VLA-JEPA's appendix. These literature differences guide diagnosis; they are not implemented changes to the current study.

## Next gates

1. Fit sixteen frames from training episode zero with native losses and fresh optimizer state. Require at least 90% lower arm-command MSE and at most 5% gripper errors on these frames. This is an overfit diagnostic, not a deliverable model.
2. Repeat under the corrected image pipeline before drawing architecture conclusions. If needed, compare explicit partial backbone trainability. Check trajectories and action traces before selecting larger compute.
3. Register a separate recovery training recipe, with backbone learning rate, batch/exposure, checkpoint selection and cost ceiling declared before running. Keep data/split, action/world architecture and inherited losses fixed.
4. Select by held-out physical arm error; use development rollouts to diagnose control. Then perform matched final success, latency and memory measurements. Export and verify durable artifacts before rental deletion.

The [registered recovery recipe](../../evaluation/recovery_config.json) starts from the verified native legacy 10k weights with fresh AdamW, batch eight and four trainable decoder layers. Decoder/other learning rates are 1e-5/1e-4, with 200-update warmup and a fixed 20k cosine horizon. Stages at 2k/5k/10k/20k preserve native optimizer, RNG and scheduler state. Every 500 updates measures native loss and physical action/gripper errors on the frozen 200 held-out frames, excluding padded actions. The latest two checkpoints and the best by arm MSE are retained; final LIBERO is reserved.

Native batch-eight training, exact FP32-master restoration, checkpoint resume and full validation scoring passed locally. Recovery training is running on one RTX 5090 at USD0.526/hour, within the original USD14 total ceiling. Cloud CUDA/BF16, LIBERO EGL reset and the frozen timing-bank hash passed. Training stops by October 2 at 18:00 UTC to allow verified export before provider cleanup at 20:00 UTC. Stage backups and exporter restart after PC boot. The goal remains open until actual task performance passes the acceptance criterion or an external resource limit prevents progress.

The plugin's current GPU call is corrected. The frozen n0008 source is copied and amended using `python -m evaluation.recovery_source`; its [manifest](diagnostics/image_pipeline_amendment.json) records the sole preprocessing correction. Neither the original RSI source nor its journal is altered. Training and evaluation with corrected inputs require new provenance and do not replace the old measurements.

Rebuild the [action-accuracy vector figure](figures/F7_action_audit.pdf) with `python -m evaluation.recovery_analysis`. Run commands are documented by `python -m evaluation.action_audit --help`, `python -m evaluation.gradient_audit --help`, and `python -m evaluation.overfit_probe --help`; checkpoints and datasets remain local runtime artifacts.
