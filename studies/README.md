# Experiment protocol

The [evaluation plan and delivery tracker](EVALUATION_PLAN.md) records the agreed experiments, budgets, five figures, summary table, and remaining work. Update it with evidence as deliverables are completed. Dream-RSI is deferred from this plan.

The main comparison is pretrained VLA-JEPA, its quantized inference variant, and VLA-JEPA with SmolVLM2-500M-Video-Instruct. Dream-RSI is optional after those comparisons work.

Next experiment direction: [replace Qwen with SmolVLM while retaining LIBERO-trained VLA-JEPA components](SMOLVLM_LIBERO_PROPOSAL.md), then adapt and evaluate the conditioning interface. Proposal only; no experiment is scheduled.

- Keep LIBERO suite, task set, episode count, seeds, action horizon, outer camera preparation, and hardware matched. Record checkpoint/code revisions and backbone-native image/token processing differences.
- Report per-task and aggregate success, warmed-up policy inference latency (median/p95), peak GPU memory, model size, and training budget. Separate simulator wall time from policy latency.
- Quantization must state precision, method, and affected modules. Removing an unused world model is an additional ablation, not evidence of quantization by itself.
- Fine-tune the smaller backbone's conditioning interface and compatible action/world modules on the fixed training split; do not train Qwen. Keep evaluation data held out.
- Dream-RSI screening uses held-out loss only. Confirm promising architectures with matched training and closed-loop LIBERO evaluation before claiming efficiency or task improvements.

Keep new compact results and provenance here; keep weights, datasets, raw logs, and runtime state outside Git. Attribute each result to its measured backbone and configuration.

The [hardware inventory](HARDWARE.md) and [later cross-architecture protocol](CROSS_HARDWARE_PLAN.md) define the future Vast.ai replication, after the local study.
