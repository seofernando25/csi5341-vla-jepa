"""SmolVLM2 variant of LeRobot's VLA-JEPA policy."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("Install LeRobot before using this plugin.") from exc

from .configuration_vla_jepa_smolvlm import VLAJEPASmolVLMConfig
from .modeling_vla_jepa_smolvlm import VLAJEPASmolVLMPolicy
from .processor_vla_jepa_smolvlm import make_vla_jepa_smolvlm_pre_post_processors

__all__ = [
    "VLAJEPASmolVLMConfig",
    "VLAJEPASmolVLMPolicy",
    "make_vla_jepa_smolvlm_pre_post_processors",
]
