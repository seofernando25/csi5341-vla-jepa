import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.processor import TransitionKey

import lerobot_policy_vla_jepa_lfm  # noqa: F401
from lerobot_policy_vla_jepa_lfm.lfm_interface import ResidualRMSMLPAdapter, _LinearAdapter
from lerobot_policy_vla_jepa_lfm.processor_vla_jepa_lfm import DropImagePadMasksProcessorStep


def test_plugin_registration():
    assert "vla_jepa_lfm" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("vla_jepa_lfm").name == "vla_jepa_lfm"


def test_residual_adapter_matches_linear_at_initialization():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    linear = _LinearAdapter(4, 8)
    residual = ResidualRMSMLPAdapter(4, 8)
    assert torch.allclose(linear(x), residual(x), atol=0, rtol=0)


def test_drop_image_padding_masks_only():
    step = DropImagePadMasksProcessorStep()
    image = torch.randn(2, 3, 16, 16)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.image": image,
            "observation.images.image_is_pad": mask,
            "observation.state": torch.randn(2, 8),
        }
    }
    out = step(transition)
    obs = out[TransitionKey.OBSERVATION]
    assert "observation.images.image" in obs
    assert "observation.images.image_is_pad" not in obs
    assert "observation.state" in obs


def test_load_compatible_weights_gguf(tmp_path):
    from unittest.mock import MagicMock

    import gguf
    import numpy as np

    from lerobot_policy_vla_jepa_lfm.modeling_vla_jepa_lfm import VLAJEPALFMPolicy

    fpath = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(str(fpath), "vla_jepa")
    writer.add_tensor("model.action_model.weight", np.ones((2, 3), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {
        "model.action_model.weight": torch.zeros((2, 3), dtype=torch.float32)
    }
    mock_policy.load_state_dict.return_value = ([], [])
    VLAJEPALFMPolicy._load_compatible_vla_jepa_weights(
        mock_policy, str(fpath), ("model.action_model.",)
    )
    loaded = mock_policy.load_state_dict.call_args[0][0]
    assert "model.action_model.weight" in loaded
    assert torch.equal(loaded["model.action_model.weight"], torch.ones((2, 3)))


def test_load_compatible_weights_gguf_mismatch(tmp_path):
    from unittest.mock import MagicMock

    import gguf
    import numpy as np
    import pytest

    from lerobot_policy_vla_jepa_lfm.modeling_vla_jepa_lfm import VLAJEPALFMPolicy

    fpath = tmp_path / "model.gguf"
    writer = gguf.GGUFWriter(str(fpath), "vla_jepa")
    writer.add_tensor("model.action_model.weight", np.ones((2, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {
        "model.action_model.weight": torch.zeros((2, 3), dtype=torch.float32)
    }
    with pytest.raises(ValueError, match="Incompatible VLA-JEPA initialization tensors"):
        VLAJEPALFMPolicy._load_compatible_vla_jepa_weights(
            mock_policy, str(fpath), ("model.action_model.",)
        )


def test_load_compatible_weights_safetensors(tmp_path):
    from unittest.mock import MagicMock

    from safetensors.torch import save_file

    from lerobot_policy_vla_jepa_lfm.modeling_vla_jepa_lfm import VLAJEPALFMPolicy

    fpath = tmp_path / "model.safetensors"
    save_file({"model.action_model.weight": torch.ones((2, 3))}, str(fpath))

    mock_policy = MagicMock()
    mock_policy.state_dict.return_value = {
        "model.action_model.weight": torch.zeros((2, 3), dtype=torch.float32)
    }
    mock_policy.load_state_dict.return_value = ([], [])
    VLAJEPALFMPolicy._load_compatible_vla_jepa_weights(
        mock_policy, str(fpath), ("model.action_model.",)
    )
    loaded = mock_policy.load_state_dict.call_args[0][0]
    assert "model.action_model.weight" in loaded
    assert torch.equal(loaded["model.action_model.weight"], torch.ones((2, 3)))
