from __future__ import annotations

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel, VLAJEPAPolicy
from lerobot.policies.vla_jepa.world_model import ActionConditionedVideoPredictor
from safetensors import safe_open
from torch import nn
from transformers import AutoModel, AutoVideoProcessor

from .configuration_vla_jepa_lfm import VLAJEPALFMConfig
from .lfm_interface import LFM25VLInterface

logger = logging.getLogger(__name__)


class VLAJEPALFMModel(VLAJEPAModel):
    """Reuse LeRobot's VLA-JEPA losses/action plumbing with an LFM vision-language interface."""

    def __init__(self, config: VLAJEPALFMConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        # Keep the attribute name `qwen` because upstream VLAJEPAModel helper methods use it.
        self.qwen = LFM25VLInterface(config)

        self.action_tokens, self.action_token_ids, self.embodied_action_token_id = self.qwen.expand_tokenizer()
        self.register_buffer(
            "_action_token_ids_t",
            torch.tensor(self.action_token_ids, dtype=torch.long),
            persistent=False,
        )
        self.action_model = VLAJEPAActionHead(config, cross_attention_dim=config.conditioning_dim)

        if config.enable_world_model:
            dtype = self.qwen._get_torch_dtype(config.torch_dtype)
            self.video_encoder = AutoModel.from_pretrained(config.jepa_encoder_name, dtype=dtype)
            self.video_processor = AutoVideoProcessor.from_pretrained(config.jepa_encoder_name)
            num_views = config.num_world_model_views
            tubelet_size = self.video_encoder.config.tubelet_size
            image_size = getattr(self.video_encoder.config, "image_size", None)
            if image_size is None:
                image_size = next(iter(config.image_features.values())).shape[-1]
            self.video_predictor = ActionConditionedVideoPredictor(
                num_frames=config.num_video_frames // tubelet_size,
                img_size=(image_size, image_size),
                patch_size=16,
                tubelet_size=1,
                embed_dim=self.video_encoder.config.hidden_size * num_views,
                action_embed_dim=config.conditioning_dim,
                predictor_embed_dim=self.video_encoder.config.hidden_size,
                depth=config.predictor_depth,
                num_heads=config.predictor_num_heads,
                mlp_ratio=config.predictor_mlp_ratio,
                num_action_tokens_per_step=config.num_action_tokens_per_timestep,
                dropout=config.predictor_dropout,
            )
        else:
            self.video_encoder = None
            self.video_processor = None
            self.video_predictor = None

        tubelet = self.video_encoder.config.tubelet_size if self.video_encoder is not None else config.jepa_tubelet_size
        prompt_steps = config.num_video_frames // tubelet - 1
        self.replace_prompt = "".join(
            token * config.num_action_tokens_per_timestep for token in self.action_tokens[:prompt_steps]
        )
        self.embodied_replace_prompt = config.embodied_action_token * config.num_embodied_action_tokens_per_instruction

    def _qwen_last_decoder_hidden(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        last_layer = self.qwen.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(hook)
        try:
            self.qwen.model.model(**inputs)
        finally:
            handle.remove()
        return self.qwen.project_hidden(captured[0])


class VLAJEPALFMPolicy(VLAJEPAPolicy):
    config_class = VLAJEPALFMConfig
    name = "vla_jepa_lfm"

    def __init__(self, config: VLAJEPALFMConfig, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.model = VLAJEPALFMModel(config)
        if config.init_from_vla_jepa:
            self._load_compatible_vla_jepa_weights(config.init_from_vla_jepa, config.init_prefixes)
        self.reset()

    def get_optim_params(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def _load_compatible_vla_jepa_weights(self, source: str, prefixes: tuple[str, ...]) -> None:
        path = Path(source)
        is_gguf = False
        if path.is_dir():
            if (path / "model.safetensors").is_file():
                model_file = path / "model.safetensors"
            else:
                ggufs = list(path.glob("*.gguf"))
                if ggufs:
                    model_file = ggufs[0]
                    is_gguf = True
                else:
                    model_file = path / "model.safetensors"
        elif path.is_file():
            model_file = path
            is_gguf = model_file.suffix == ".gguf"
        else:
            try:
                model_file = Path(hf_hub_download(repo_id=source, filename="model.safetensors"))
            except Exception:
                from huggingface_hub import HfApi
                api = HfApi()
                repo_files = api.list_repo_files(repo_id=source)
                ggufs = [f for f in repo_files if f.endswith(".gguf")]
                if ggufs:
                    gguf_name = "vla-jepa.gguf" if "vla-jepa.gguf" in ggufs else ggufs[0]
                    model_file = Path(hf_hub_download(repo_id=source, filename=gguf_name))
                    is_gguf = True
                else:
                    raise

        current = self.state_dict()
        selected: dict[str, torch.Tensor] = {}
        mismatched: list[str] = []

        if is_gguf:
            import gguf

            reader = gguf.GGUFReader(str(model_file))
            ah_proj = {
                "ah.act_enc.l1.weight": "model.action_model.action_encoder.layer1.weight",
                "ah.act_enc.l1.bias": "model.action_model.action_encoder.layer1.bias",
                "ah.act_enc.l2.weight": "model.action_model.action_encoder.layer2.weight",
                "ah.act_enc.l2.bias": "model.action_model.action_encoder.layer2.bias",
                "ah.act_enc.l3.weight": "model.action_model.action_encoder.layer3.weight",
                "ah.act_enc.l3.bias": "model.action_model.action_encoder.layer3.bias",
                "ah.state_enc.l1.weight": "model.action_model.state_encoder.layer1.weight",
                "ah.state_enc.l1.bias": "model.action_model.state_encoder.layer1.bias",
                "ah.state_enc.l2.weight": "model.action_model.state_encoder.layer2.weight",
                "ah.state_enc.l2.bias": "model.action_model.state_encoder.layer2.bias",
                "ah.act_dec.l1.weight": "model.action_model.action_decoder.layer1.weight",
                "ah.act_dec.l1.bias": "model.action_model.action_decoder.layer1.bias",
                "ah.act_dec.l2.weight": "model.action_model.action_decoder.layer2.weight",
                "ah.act_dec.l2.bias": "model.action_model.action_decoder.layer2.bias",
                "ah.time_emb.l1.weight": "model.action_model.model.timestep_encoder.timestep_embedder.linear_1.weight",
                "ah.time_emb.l1.bias": "model.action_model.model.timestep_encoder.timestep_embedder.linear_1.bias",
                "ah.time_emb.l2.weight": "model.action_model.model.timestep_encoder.timestep_embedder.linear_2.weight",
                "ah.time_emb.l2.bias": "model.action_model.model.timestep_encoder.timestep_embedder.linear_2.bias",
                "ah.proj_out1.weight": "model.action_model.model.proj_out_1.weight",
                "ah.proj_out1.bias": "model.action_model.model.proj_out_1.bias",
                "ah.proj_out2.weight": "model.action_model.model.proj_out_2.weight",
                "ah.proj_out2.bias": "model.action_model.model.proj_out_2.bias",
                "ah.future_tokens": "model.action_model.future_tokens.weight",
                "ah.pos_embd": "model.action_model.position_embedding.weight",
            }
            dit_map = {
                "adaln.weight": "norm1.linear.weight",
                "adaln.bias": "norm1.linear.bias",
                "attn_q.weight": "attn1.to_q.weight",
                "attn_q.bias": "attn1.to_q.bias",
                "attn_k.weight": "attn1.to_k.weight",
                "attn_k.bias": "attn1.to_k.bias",
                "attn_v.weight": "attn1.to_v.weight",
                "attn_v.bias": "attn1.to_v.bias",
                "attn_o.weight": "attn1.to_out.0.weight",
                "attn_o.bias": "attn1.to_out.0.bias",
                "ff0.weight": "ff.net.0.proj.weight",
                "ff0.bias": "ff.net.0.proj.bias",
                "ff2.weight": "ff.net.2.weight",
                "ff2.bias": "ff.net.2.bias",
            }
            for tensor in reader.tensors:
                key = tensor.name
                candidate_keys = [key]
                if key in ah_proj:
                    candidate_keys.append(ah_proj[key])
                elif key.startswith("ah.dit."):
                    parts = key.split(".")
                    layer_idx = parts[2]
                    sub = ".".join(parts[3:])
                    if sub in dit_map:
                        candidate_keys.append(
                            f"model.action_model.model.transformer_blocks.{layer_idx}.{dit_map[sub]}"
                        )
                if not key.startswith("model."):
                    candidate_keys.append("model." + key)
                for k in candidate_keys:
                    if not any(k.startswith(p) for p in prefixes):
                        continue
                    if k not in current:
                        continue
                    raw = torch.from_numpy(tensor.data.copy())
                    if raw.dtype == torch.uint8:
                        pt_shape = tuple(int(x) for x in reversed(tensor.shape))
                        t = raw.view(torch.bfloat16).reshape(pt_shape)
                    else:
                        t = raw
                    if t.shape != current[k].shape:
                        mismatched.append(f"{k}: {tuple(t.shape)} != {tuple(current[k].shape)}")
                        continue
                    selected[k] = t
        else:
            with safe_open(model_file, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
                    if not any(key.startswith(p) for p in prefixes):
                        continue
                    if key not in current:
                        continue
                    tensor = handle.get_tensor(key)
                    if tensor.shape != current[key].shape:
                        mismatched.append(f"{key}: {tuple(tensor.shape)} != {tuple(current[key].shape)}")
                        continue
                    selected[key] = tensor

        if mismatched:
            raise ValueError("Incompatible VLA-JEPA initialization tensors:\n" + "\n".join(mismatched))
        if not selected:
            logger.warning(
                "No compatible tensors found in %s for prefixes %s; keeping default initialization",
                model_file,
                prefixes,
            )
            return
        _missing, unexpected = self.load_state_dict(selected, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected initialization keys: {unexpected}")
        logger.info("Loaded %d compatible tensors from %s", len(selected), source)
