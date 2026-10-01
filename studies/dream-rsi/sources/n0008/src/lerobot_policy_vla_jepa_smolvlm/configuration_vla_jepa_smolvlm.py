from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig


@PreTrainedConfig.register_subclass("vla_jepa_smolvlm")
@dataclass
class VLAJEPASmolVLMConfig(VLAJEPAConfig):
    """VLA-JEPA with SmolVLM2 replacing Qwen3-VL.

    The action head and JEPA predictor keep their 2048-D conditioning interface. The backbone's native
    decoder state is projected through a small adapter.
    """

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    conditioning_dim: int = 2048
    adapter_type: str = "linear"  # linear | residual_mlp
    decoder_adaptation: str = "penultimate_residual"  # penultimate_residual | none
    image_processor_backend: str = "pil"  # Preserve historical checkpoint preprocessing.

    # SmolVLM stays frozen by default. These switches exist for explicit later ablations.
    unfreeze_last_n: int = 0
    train_multimodal_projector: bool = False
    freeze_vision_tower: bool = True

    # Transfer only architecture-compatible VLA-JEPA modules from the official LeRobot checkpoint.
    init_from_vla_jepa: str | None = "lerobot/VLA-JEPA-Pretrain"
    init_prefixes: tuple[str, ...] = ("model.action_model.", "model.video_predictor.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.adapter_type not in {"linear", "residual_mlp"}:
            raise ValueError("adapter_type must be 'linear' or 'residual_mlp'")
        if self.decoder_adaptation not in {"penultimate_residual", "none"}:
            raise ValueError("decoder_adaptation must be 'penultimate_residual' or 'none'")
        if self.image_processor_backend not in {"pil", "torchvision"}:
            raise ValueError("image_processor_backend must be 'pil' or 'torchvision'")
        if self.conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")
        if self.unfreeze_last_n < 0:
            raise ValueError("unfreeze_last_n must be nonnegative")
        if self.freeze_qwen:
            raise ValueError(
                "freeze_qwen is a Qwen-only option; use the SmolVLM trainability fields"
            )
