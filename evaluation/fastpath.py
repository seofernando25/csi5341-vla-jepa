"""CUDA-graph execution of the VLA-JEPA action-chunk prediction at batch size 1.

The stock path launches thousands of small kernels per prediction and synchronizes with the host
inside the vision tower and the decoder mask code, so it is launch-bound. This path precomputes
everything that depends only on the instruction (token ids, rotary tables, vision index tables),
composes the same sub-modules without host synchronization, and replays one captured graph per
instruction. Noise is drawn eagerly with the stock call, so a seeded prediction uses the same
flow-matching noise as the stock path.

Contract: one observation per call, two equally sized camera images, no padding in the prompt,
and a state vector. With `use_compile=False` and NF4 weights the outputs equal the stock outputs
bit for bit; `torch.compile` fuses kernels and changes bf16 rounding.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
from transformers.image_utils import SizeDict

from evaluation import nf4


@dataclass
class Plan:
    """Instruction-dependent constants plus the static buffers and graph for one prompt shape."""

    text_embeds: torch.Tensor
    text_position_ids: torch.Tensor
    rope: tuple
    visual_index: torch.Tensor
    action_index: torch.Tensor
    vision_position: torch.Tensor
    vision_rope: tuple
    lengths: list
    images: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None


def _vision_attention(
    self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs
):
    """Vision attention with host-known sequence lengths, so no `.tolist()` synchronization."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    attend = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    lengths = self.static_lengths or (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    chunks = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
    outputs = [
        attend(
            self, a, b, c, attention_mask=None, scaling=self.scaling, dropout=0.0, is_causal=False
        )[0]
        for a, b, c in zip(*chunks)
    ]
    return self.proj(torch.cat(outputs, dim=1).reshape(length, -1).contiguous())


def _sinusoid(self, timesteps):
    """Sinusoidal embedding with its constants built on the device, so it is capturable."""
    timesteps = timesteps.float()
    batch_size, seq_len = timesteps.shape
    half_dim = self.embedding_dim // 2
    exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
    exponent = exponent * (self.log_base / max(half_dim, 1))
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).view(batch_size, seq_len, -1)


def _nf4_linear(self, x):
    weight = nf4.dequantize(self.weight.data, self.tables, self.shape)
    return torch.nn.functional.linear(x, weight, self.bias)


def _dense_linear(self, x):
    return torch.nn.functional.linear(x, self.dense_weight, self.bias)


class GraphedPolicy:
    """Drop-in `predict_action_chunk` for a loaded B16 or Q4 `VLAJEPAPolicy`, one graph per prompt."""

    def __init__(self, policy, use_compile=True, dense=True):
        """`dense=True` keeps NF4-rounded weights resident in bf16 beside the packed ones (about
        3 GB more for the pruned model), which removes the per-call dequantization. `dense=False`
        keeps only the packed weights and dequantizes each linear just before its GEMM."""
        self.policy = policy
        self.use_compile = use_compile
        self.dense = dense
        self.model = policy.model
        qwen = self.model.qwen.model.model
        self.vision, self.decoder, self.head = (
            qwen.visual,
            qwen.language_model,
            self.model.action_model,
        )
        self.processor = self.model.qwen.processor.image_processor
        device = next(policy.parameters()).device
        shape = (1, 3, 1, 1)
        self.mean = torch.tensor(
            self.processor.image_mean, dtype=torch.float32, device=device
        ).view(shape)
        self.std = torch.tensor(self.processor.image_std, dtype=torch.float32, device=device).view(
            shape
        )
        self.plans: dict[str, Plan] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self._prepare_nf4()
        for module in self.head.modules():
            if type(module).__name__ == "SinusoidalPositionalEncoding":
                module.log_base = torch.log(torch.tensor(10000.0, device=device))
                module.forward = types.MethodType(_sinusoid, module)
        self.attentions = [b.attn for b in self.vision.blocks if hasattr(b, "attn")]
        for attn in self.attentions:
            attn.static_lengths = None
            attn.forward = types.MethodType(_vision_attention, attn)

    # -- NF4 linears anywhere in the model: dequantize per call (or once), then GEMM --

    def _prepare_nf4(self):
        import bitsandbytes as bnb

        for linear in self.model.modules():
            if not isinstance(linear, bnb.nn.Linear4bit):
                continue
            linear.shape = tuple(linear.weight.quant_state.shape)
            linear.tables = nf4.tables(linear.weight)
            if self.dense:
                linear.dense_weight = nf4.dequantize(
                    linear.weight.data, linear.tables, linear.shape
                )
                linear.forward = types.MethodType(_dense_linear, linear)
            else:
                linear.forward = types.MethodType(_nf4_linear, linear)

    # -- plan construction (once per instruction, eager) --

    @torch.inference_mode()
    def _plan(self, instruction, images, state) -> Plan:
        model = self.model
        qwen_inputs = model.qwen.build_inputs(
            [images], [instruction], model.replace_prompt, model.embodied_replace_prompt
        )
        grid = qwen_inputs["image_grid_thw"]
        captured = {}

        def grab(_, __, kwargs):
            captured.update(kwargs)

        handle = self.decoder.register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.qwen.model.model(**qwen_inputs)
        finally:
            handle.remove()
        ids = qwen_inputs["input_ids"]
        if ids.shape[0] != 1 or not bool(qwen_inputs["attention_mask"].all()):
            raise ValueError("Graphed execution requires one unpadded prompt per call")
        merged = self.vision.spatial_merge_size**2
        lengths = [int(t * h * w) for t, h, w in grid.tolist()]
        for attn in self.attentions:
            attn.static_lengths = lengths
        embeds = captured["inputs_embeds"]
        position_ids = captured["position_ids"]
        emb = self.vision.rot_pos_emb(grid).reshape(sum(lengths), -1)
        emb = torch.cat((emb, emb), dim=-1)
        assert sum(lengths) // merged == int(captured["visual_pos_masks"].sum())
        return Plan(
            text_embeds=embeds.clone(),
            text_position_ids=position_ids[0] if position_ids.shape[0] == 4 else None,
            rope=self.decoder.rotary_emb(embeds, position_ids),
            visual_index=captured["visual_pos_masks"][0].nonzero().squeeze(1),
            action_index=(ids[0] == model.embodied_action_token_id).nonzero().squeeze(1),
            vision_position=self.vision.fast_pos_embed_interpolate(grid),
            vision_rope=(emb.cos(), emb.sin()),
            lengths=lengths,
            images=torch.stack(images).clone(),
            state=state.clone(),
            noise=torch.zeros(
                1, self.head.action_horizon, self.head.config.action_dim, device=state.device
            ),
        )

    # -- the graphed computation --

    def _pixels(self, plan):
        ip = self.processor
        side = round(plan.lengths[0] ** 0.5) * ip.patch_size
        x = ip.resize(
            image=plan.images, size=SizeDict(height=side, width=side), resample=ip.resample
        )
        x = (x - self.mean).div_(self.std)
        n, c, height, width = x.shape
        patch, merge, temporal = ip.patch_size, ip.merge_size, ip.temporal_patch_size
        x = x.unsqueeze(1).expand(n, temporal, c, height, width)
        x = x.reshape(
            n,
            1,
            temporal,
            c,
            height // patch // merge,
            merge,
            patch,
            width // patch // merge,
            merge,
            patch,
        )
        return x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(
            n * (height // patch) * (width // patch), -1
        )

    def _body(self, plan):
        vision, decoder = self.vision, self.decoder
        pixels = self._pixels(plan).to(vision.dtype)
        hidden = vision.patch_embed(pixels) + plan.vision_position
        deep = []
        for number, block in enumerate(vision.blocks):
            hidden = block(hidden, cu_seqlens=None, position_embeddings=plan.vision_rope)
            if number in vision.deepstack_visual_indexes:
                position = vision.deepstack_visual_indexes.index(number)
                deep.append(vision.deepstack_merger_list[position](hidden))
        image_embeds = vision.merger(hidden)
        hidden = plan.text_embeds.clone()
        hidden[0, plan.visual_index] = image_embeds.to(hidden.dtype)
        for number, layer in enumerate(decoder.layers):
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=plan.text_position_ids,
                position_embeddings=plan.rope,
            )
            if number < len(deep):
                hidden = hidden.clone()
                hidden[0, plan.visual_index] = hidden[0, plan.visual_index] + deep[number].to(
                    hidden.dtype
                )
        tokens = hidden[:, plan.action_index].float()
        return self._flow(tokens, plan)

    def _flow(self, tokens, plan):
        head = self.head
        actions = plan.noise
        steps = head.num_inference_timesteps
        for step in range(steps):
            value = int(step / steps * head.config.action_num_timestep_buckets)
            timesteps = torch.full((1,), value, device=tokens.device, dtype=torch.long)
            hidden = head._build_inputs(actions, plan.state.float(), timesteps)
            pred = head.model(
                hidden_states=hidden, encoder_hidden_states=tokens, timestep=timesteps
            )
            actions = actions + (1.0 / steps) * head.action_decoder(pred[:, -head.action_horizon :])
        return actions

    @torch.inference_mode()
    def _capture(self, plan):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            body = torch.compile(self._body) if self.use_compile else self._body
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    body(plan)
            torch.cuda.current_stream().wait_stream(warmup)
            plan.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(plan.graph, pool=self.pool):
                plan.output = body(plan)

    # -- public --

    @torch.inference_mode()
    def predict_action_chunk(self, batch, noise=None):
        policy = self.policy
        inputs = policy._prepare_model_inputs(batch, training=False)
        images, instruction = inputs["images"][0], inputs["instructions"][0]
        state = inputs["state"]
        plan = self.plans.get(instruction)
        if plan is None:
            plan = self.plans[instruction] = self._plan(instruction, images, state)
            self._capture(plan)
        plan.images.copy_(torch.stack(images))
        plan.state.copy_(state)
        plan.noise.copy_(
            torch.randn(
                1,
                plan.noise.shape[1],
                plan.noise.shape[2],
                dtype=torch.float32,
                device=plan.noise.device,
            )
        )
        plan.graph.replay()
        return plan.output.clone()


def attach(policy, **options):
    """Route `policy.predict_action_chunk` (and so `select_action`) through graphed execution."""
    graphed = GraphedPolicy(policy, **options)
    policy.predict_action_chunk = graphed.predict_action_chunk
    policy.graphed = graphed
    return {"engine": "cuda_graph", **options}
