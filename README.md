# CSI 5341 — Efficient VLA-JEPA

Course project by Noah Sprenger and Fernando Nogueira: evaluate how to reduce VLA-JEPA's hardware requirements while preserving robotic manipulation performance.

| Comparison | Scope |
| --- | --- |
| Baseline | Evaluate pretrained VLA-JEPA in LIBERO. |
| Quantized inference | Compare reduced precision against the same baseline. |
| Smaller backbone | Replace Qwen with SmolVLM2-500M and run reduced fine-tuning. |
| Dream-RSI (optional) | Search for better SmolVLM conditioning architectures. |

Report task success, inference latency, peak GPU memory, and hardware under matched evaluation settings. Full training from scratch is outside scope. Track experiments, figures, and remaining deliverables in the [evaluation plan](studies/EVALUATION_PLAN.md).

[Mac / next-agent handoff](HANDOFF.md) · [Working report](studies/evaluation/report/report.pdf) · [Reproduction commands](studies/evaluation/README.md) · [Hardware](studies/HARDWARE.md) · [Later cloud comparison](studies/CROSS_HARDWARE_PLAN.md)

## Presentation

The [immersive presentation](web/presentation/README.md) includes its required local footage, narration, fonts, diagrams and 3D dependencies. Serve the repository with `python3 -m http.server 8000 --bind 127.0.0.1` and open `http://localhost:8000/web/presentation/`.

[Slides](web/presentation/documents/slides.pdf) · [Slides with backups](web/presentation/documents/slides-with-backups.pdf) · [Presenter guide](web/presentation/documents/presenter-guide.pdf)

## Setup

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra dev --extra eval
PYTHONPATH=src uv run pytest
PYTHONPATH=src uv run python scripts/smoke_plugin.py
PYTHONPATH=src uv run python -m rsi dry-run
```

The active plugin is `vla_jepa_smolvlm`, using [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct). The backbone swap requires fine-tuning; it is not a ready-trained robot policy. Quantization remains a separate comparison arm.

[Dream-RSI commands](rsi/README.md) operate one fresh, resumable SmolVLM study. New results belong in [studies/](studies/README.md).

References: [VLA-JEPA](https://arxiv.org/abs/2602.10098), [V-JEPA 2](https://arxiv.org/abs/2506.09985), [SmolVLM](https://arxiv.org/abs/2504.05299), [Dream-RSI](https://arxiv.org/abs/2609.14858). Built on Hugging Face LeRobot; no weights or datasets are redistributed.
