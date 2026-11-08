"""Outer LIBERO processing around a policy, and the fixed observation bank used for timing."""

import torch

from evaluation.common import ROOT
from evaluation.simulator import numpy_raw
from evaluation.sources import artifact

BANK = ROOT / "outputs/evaluation/observations.pt"


def observation_bank():
    """100 settled development observations (10 per task), written by `run collect`."""
    return torch.load(BANK, weights_only=True)


def inference_autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class Pipeline:
    """Simulator observation → policy batch, and policy action → simulator command."""

    def __init__(self, policy, checkpoint=None):
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies import make_pre_post_processors

        self.env_pre, _ = LiberoEnv(task="libero_spatial").get_env_processors()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint or artifact("baseline"),
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )

    def reset(self):
        self.env_pre.reset()
        self.pre.reset()
        self.post.reset()

    def prepare(self, raw, instruction):
        """Batch one batched raw observation; processor state carries across an episode."""
        from lerobot.envs import preprocess_observation

        observation = preprocess_observation(raw)
        observation["task"] = [instruction]
        return self.pre(self.env_pre(observation))

    def sample(self, sample):
        """Fresh processor state, then the batch for one observation-bank sample."""
        self.reset()
        return self.prepare(numpy_raw(sample["raw"]), sample["instruction"])
