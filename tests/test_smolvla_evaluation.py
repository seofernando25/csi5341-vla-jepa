from types import SimpleNamespace
import torch
from evaluation.run import inference_autocast


def test_smolvla_preserves_published_disabled_amp():
    policy = SimpleNamespace(config=SimpleNamespace(type="smolvla", use_amp=False))
    with inference_autocast(policy):
        assert not torch.is_autocast_enabled("cuda")
