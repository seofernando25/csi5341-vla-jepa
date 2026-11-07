"""Check compression math, attention geometry and selection without robot rollouts."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.compression.transforms import (
    CompactEmbedding,
    Int8Embedding,
    SkipBlock,
    independent_heads,
    keep_channels,
    prune_mask,
    slice_linear,
    storage_inventory,
)


@pytest.mark.parametrize("gemm_backend", [None, "triton"])
def test_compiled_linears_preserve_payload_hooks_and_module_binding(monkeypatch, gemm_backend):
    from evaluation.compression import compiled
    from evaluation.compression.artifacts import tensor_fingerprint

    model = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2, bias=False)).eval()
    first = torch.randn(2, 3)
    second = torch.randn(4, 5)
    expected = [model[0](first).detach(), model[1](second).detach()]
    payload = tensor_fingerprint(model)
    parameters = [id(value) for value in model.parameters()]
    hooks = []
    model[0].register_forward_hook(lambda module, args, output: hooks.append(output.shape))
    monkeypatch.setattr(
        compiled,
        "selected_linears",
        lambda policy, scope: {str(i): (model, str(i), layer) for i, layer in enumerate(model)},
    )
    options = []
    actual_compile = torch.compile

    def cpu_graph(operation, **kwargs):
        options.append(kwargs)
        return actual_compile(operation, backend="eager", fullgraph=True, dynamic=True)

    monkeypatch.setattr(torch, "compile", cpu_graph)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    monkeypatch.setattr(compiled.inductor_config, "max_autotune_gemm_backends", "ATEN,TRITON")
    change = {"scope": "text", **({"gemm_backend": gemm_backend} if gemm_backend else {})}
    event = compiled.compile_linears(model, change)
    torch.testing.assert_close(model[0](input=first), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(model[1](second), expected[1], rtol=0, atol=0)
    assert hooks == [torch.Size([2, 5])]
    assert parameters == [id(value) for value in model.parameters()]
    assert tensor_fingerprint(model) == payload
    assert options == [
        {
            "backend": "inductor",
            "mode": "max-autotune-no-cudagraphs" if gemm_backend else "default",
            "fullgraph": True,
            "dynamic": True,
        }
    ]
    if gemm_backend:
        assert event["gemm_backend"] == "triton"
        assert event["inductor_settings"]["max_autotune_gemm_backends"] == "TRITON"
    assert event["silent_eager_fallback"] is False
    with pytest.raises(ValueError, match="already applied"):
        compiled.compile_linears(model, {"scope": "text"})


def test_compiled_linears_reject_unknown_backend():
    from evaluation.compression.compiled import compile_linears

    with pytest.raises(ValueError, match="Unknown compiled GEMM backend"):
        compile_linears(None, {"scope": "action", "gemm_backend": "unknown"})


def test_compiled_linear_backend_errors_are_propagated(monkeypatch):
    from evaluation.compression import compiled

    model = nn.Linear(3, 2)
    monkeypatch.setattr(
        compiled, "selected_linears", lambda policy, scope: {"linear": (model, "", model)}
    )

    def failing_backend(operation, **kwargs):
        def fail(*args):
            raise RuntimeError("compiler failure")

        return fail

    monkeypatch.setattr(torch, "compile", failing_backend)
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    compiled.compile_linears(model, {"scope": "text"})
    with pytest.raises(RuntimeError, match="compiler failure"):
        model(torch.randn(1, 3))


def test_compiled_screen_shape_priming_resets_each_observation_seed(monkeypatch):
    from contextlib import nullcontext

    from evaluation.compression import study

    bank = [{"index": index} for index in range(20)]
    calls, resets = [], []

    class Policy:
        def reset(self):
            resets.append(True)

        def predict_action_chunk(self, index):
            calls.append((index, torch.initial_seed()))
            return torch.ones(1, 7, 7)

    monkeypatch.setattr(study, "inputs", lambda sample, pipeline: sample["index"])
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    result = study.prime_screen_shapes(Policy(), bank, None)
    assert calls == [(index, study.PROTOCOL["seed"] + index) for index in range(20)]
    assert len(resets) == len(bank)
    assert result["observations"] == len(bank)
    assert result["included_in_diagnostic_rows"] is False
    assert result["peak_allocated_bytes"] == 100
    assert result["peak_reserved_bytes"] == 200


def test_wanda_uses_activation_energy_and_exact_per_row_counts():
    w = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    scale = torch.tensor([100.0, 1.0, 1.0, 1.0])
    mask = prune_mask(w, 0.5, scale)
    assert mask.sum(1).tolist() == [2, 2]
    assert not mask[:, 0].any()
    assert torch.equal(mask, torch.tensor([[False, True, True, False], [False, False, True, True]]))


def test_2of4_masks_every_group():
    w = torch.arange(24).float().reshape(3, 8)
    mask = prune_mask(w, 0.5, pattern="2:4")
    assert (mask.reshape(3, 2, 4).sum(-1) == 2).all()
    with pytest.raises(ValueError):
        prune_mask(torch.ones(2, 7), 0.5, pattern="2:4")


def test_channel_compaction_matches_zeroed_dense_network():
    torch.manual_seed(9)
    a, b = nn.Linear(8, 16), nn.Linear(16, 5)
    importance = torch.arange(16).float() + 1
    keep = keep_channels(b, 0.5, importance, multiple=4)
    x = torch.randn(3, 8)
    dense_hidden = torch.nn.functional.gelu(a(x))
    masked = dense_hidden.clone()
    dropped = torch.ones(16, dtype=torch.bool)
    dropped[keep] = False
    masked[:, dropped] = 0
    compact = slice_linear(b, columns=keep)(torch.nn.functional.gelu(slice_linear(a, rows=keep)(x)))
    assert torch.allclose(compact, b(masked), rtol=1e-5, atol=1e-6)
    assert compact.shape == (3, 5)


def test_embedding_row_quantization_error_and_real_storage_reduction():
    torch.manual_seed(42)
    source = nn.Embedding(100, 64)
    q = Int8Embedding(source)
    ids = torch.tensor([[0, 3, 99]])
    assert q.values.dtype == torch.int8
    assert (q(ids) - source(ids)).abs().max() < 0.03
    assert (
        storage_inventory(q)["resident_tensor_storage_bytes"]
        < storage_inventory(source)["resident_tensor_storage_bytes"] / 2
    )


def test_compact_embedding_preserves_rows_and_rejects_unseen_ids():
    source = nn.Embedding(100, 4)
    compact = CompactEmbedding(source, torch.tensor([99, 3, 3, 0]))
    ids = torch.tensor([[99, 0], [3, 99]])
    assert torch.equal(compact(ids), source(ids))
    assert compact.num_embeddings == 3
    with pytest.raises(ValueError, match="Unseen"):
        compact(torch.tensor([2]))


def test_residual_bypass_accepts_each_transformer_call_signature():
    x = torch.randn(2, 4, 8)
    assert SkipBlock()(x, temb=torch.ones(2, 8)) is x
    assert SkipBlock()(hidden_states=x, position_embeddings=None) is x


def test_storage_deduplicates_tied_weights():
    m = nn.Module()
    m.a = nn.Linear(8, 8, bias=False)
    m.b = nn.Linear(8, 8, bias=False)
    m.b.weight = m.a.weight
    assert storage_inventory(m)["resident_tensor_storage_bytes"] == 8 * 8 * 4


def test_affine_int4_includes_zero_and_bounds_reconstruction_error():
    from evaluation.compression.advanced import affine_quantize

    torch.manual_seed(11)
    weight = torch.randn(8, 256)
    weight[0, :128] = 0
    values, scales, zeros, reconstructed = affine_quantize(weight)
    assert values.dtype == torch.uint8
    assert values.max() <= 15
    assert torch.isfinite(scales).all()
    assert torch.equal(reconstructed[0, :128], weight[0, :128])
    error = (reconstructed - weight).abs().reshape(8, 2, 128)
    assert (error <= scales[..., None] / 2 + 1e-6).all()
    assert ((zeros >= 0) & (zeros <= 15)).all()


def test_linear_cka_is_invariant_to_orthogonal_rotation_and_translation():
    from evaluation.compression.cka import linear_cka

    torch.manual_seed(4)
    x = torch.randn(32, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    assert linear_cka(x, x @ rotation + 10) == pytest.approx(1, abs=1e-6)
    assert linear_cka(x, x[torch.randperm(len(x))]) < 0.6


def test_attention_matching_ratios_use_published_clamps_and_neutral_ranges():
    from evaluation.compression.attention_matching import alpha_ratio, beta_ratio

    alpha = alpha_ratio(torch.tensor([1.01, 10.0, 0.1, 1.0]), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(alpha, torch.tensor([1.0, 1.4, 0.7, 1.0]))
    assert beta_ratio(1.01, 1) == 1
    assert beta_ratio(100, 1) == pytest.approx(torch.exp(torch.tensor(0.30)).item())
    assert beta_ratio(1, 100) == pytest.approx(torch.exp(torch.tensor(-0.30)).item())


def test_packed_archive_keeps_quantized_embedding_metadata_and_tied_aliases():
    from evaluation.compression.artifacts import packed_state

    m = nn.Module()
    m.embedding = Int8Embedding(nn.Embedding(50, 16))
    m.a = nn.Linear(16, 4, bias=False)
    m.b = nn.Linear(16, 4, bias=False)
    m.b.weight = m.a.weight
    tensors, layout = packed_state(m)
    assert tensors["embedding.values"].dtype == torch.int8
    assert tensors["embedding.scales"].shape == (50,)
    assert layout["aliases"] == {"b.weight": "a.weight"}
    assert "b.weight" not in tensors


def test_structural_action_heads_match_dense_masked_head_outputs():
    import copy

    from diffusers.models.attention import Attention

    torch.manual_seed(9)
    dense = Attention(query_dim=16, heads=4, dim_head=4, cross_attention_dim=12, bias=True)
    compact = copy.deepcopy(dense)
    importance = torch.arange(16).float() + 1
    scores = (dense.to_out[0].weight.abs().sum(0) * importance.sqrt()).reshape(4, 4).sum(-1)
    kept = scores.argsort(descending=True)[:2].sort().values
    dropped = torch.ones(4, dtype=torch.bool)
    dropped[kept] = False
    dense.to_out[0].weight.data[:, dropped.repeat_interleave(4)] = 0
    independent_heads(compact, 0.5, importance)
    x, context = torch.randn(2, 7, 16), torch.randn(2, 11, 12)
    assert torch.allclose(
        compact(x, encoder_hidden_states=context),
        dense(x, encoder_hidden_states=context),
        rtol=1e-5,
        atol=1e-6,
    )
    assert compact.heads == 2
    assert compact.to_q.out_features == compact.to_out[0].in_features == 8


def test_structural_vision_heads_match_dense_masked_head_outputs():
    import copy

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

    torch.manual_seed(9)
    config = Qwen3VLVisionConfig(hidden_size=32, num_heads=4)
    config._attn_implementation = "sdpa"
    dense = Qwen3VLVisionAttention(config)
    compact = copy.deepcopy(dense)
    importance = torch.arange(32).float() + 1
    scores = (dense.proj.weight.abs().sum(0) * importance.sqrt()).reshape(4, 8).sum(-1)
    kept = scores.argsort(descending=True)[:2].sort().values
    dropped = torch.ones(4, dtype=torch.bool)
    dropped[kept] = False
    dense.proj.weight.data[:, dropped.repeat_interleave(8)] = 0
    independent_heads(compact, 0.5, importance, vision=True)
    x = torch.randn(12, 32)
    positions = (torch.ones(12, 8), torch.zeros(12, 8))
    cu = torch.tensor([0, 6, 12])
    assert torch.allclose(
        compact(x, cu, position_embeddings=positions),
        dense(x, cu, position_embeddings=positions),
        rtol=1e-5,
        atol=1e-6,
    )
    assert compact.num_heads == 2
    assert compact.qkv.out_features == 3 * compact.proj.in_features == 48


def test_development_selection_uses_smoke_success_and_preserves_frozen_choice(
    tmp_path, monkeypatch
):
    from evaluation.common import write_json
    from evaluation.compression import mission

    monkeypatch.setattr(mission, "STUDY", tmp_path)
    write_json(tmp_path / "execution_plan.json", {"protocol": "test"})

    def result(candidate, phase):
        assert phase != "final"
        return {"status": "completed", "summary": {"successes": 8}}

    monkeypatch.setattr(mission, "result", result)

    def row(identity, family, success, memory, method):
        return {
            "candidate": identity,
            "family": family,
            "changes": [{"method": method, "scope": "text_mlp"}],
            "phases": {
                "screen": {
                    "status": "completed",
                    "summary": {"peak_allocated_bytes": memory, "pipeline_median_ms": 1},
                },
                "smoke": {"status": "completed", "summary": {"successes": success}},
                "final": {"status": "completed", "summary": {"successes": 0}},
            },
        }

    rows = [
        row("joint", "joint compression", 8, 50, "bnb"),
        row("pruned", "channels", 7, 80, "channels"),
        row("bad", "channels", 6, 10, "channels"),
    ]
    monkeypatch.setattr(mission, "ledger", lambda: rows)
    execution = {"development_smoke_tolerance_episodes": 1, "development_maximum_candidates": 2}
    assert mission.development_selection(execution) == ["B16", "joint", "pruned"]
    assert (
        "bad"
        not in json.loads((tmp_path / "development_selection.json").read_text())[
            "eligible_evidence"
        ]
    )
    # Later evidence never silently rewrites the recorded selection.
    rows.clear()
    assert mission.development_selection(execution) == ["B16", "joint", "pruned"]


def test_structural_boundary_probes_use_real_outcomes_and_separate_rankings():
    from evaluation.compression.boundaries import select

    def row(identity, count, success=None, criterion="default", screen="completed"):
        phases = {
            "screen": {"status": screen, "summary": {"mean_physical_arm_change_mse": 0}},
            "final": {"status": "completed", "summary": {"successes": 500}},
        }
        if success is not None:
            phases["smoke"] = {"status": "completed", "summary": {"successes": success}}
        return {
            "candidate": identity,
            "changes": [
                {"method": "layers", "scope": "text", "count": count, "criterion": criterion}
            ],
            "phases": phases,
        }

    rows = [
        row("mild", 1, 8),
        row("mid", 4),
        row("upper-mid", 8),
        row("strong", 14, 2),
        row("runtime-failure", 7, screen="failed"),
        row("cka", 4, criterion="cka"),
    ]
    selected, evidence = select(rows, 7)
    assert selected == ["cka", "upper-mid"]
    assert "runtime-failure" not in selected
    assert evidence["layers/text/default"]["completed_smokes"]["strong"]["successes"] == 2
    # A failed midpoint moves the next probe down; final outcomes and zero
    # diagnostic error neither qualify a configuration nor guide this choice.
    rows[2]["phases"]["smoke"] = {"status": "completed", "summary": {"successes": 6}}
    selected, _ = select(rows, 7)
    assert selected == ["cka", "mid"]
    rows[2]["phases"]["smoke"]["summary"]["successes"] = 8
    rows[0]["phases"]["smoke"]["summary"]["successes"] = 6
    _, evidence = select(rows, 7)
    assert evidence["layers/text/default"]["nonmonotonic_observed_outcomes"]


def test_family_endpoint_schedule_keeps_families_beyond_the_old_cap(monkeypatch):
    from evaluation.compression import mission

    identities = [f"family-{index:02}" for index in range(47)]

    def result(candidate, phase):
        assert phase != "final"
        if phase == "smoke":
            return None
        return {
            "status": "completed",
            "candidate": {
                "id": candidate,
                "changes": [{"method": "bnb", "scope": candidate, "bits": 4}],
            },
            "summary": {"peak_allocated_bytes": 1},
        }

    monkeypatch.setattr(mission, "result", result)
    assert mission.additional_smokes({"candidates": identities}) == identities


@pytest.mark.parametrize("fraction", [0, 0.5, 0.75])
def test_visual_compaction_preserves_both_views_positions_fusion_and_conditioning(fraction):
    from evaluation.compression.tokens import visual_token_pruning

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])

        def forward(
            self,
            inputs_embeds,
            visual_pos_masks,
            position_ids,
            attention_mask,
            deepstack_visual_embeds,
        ):
            self.compact_positions = position_ids.clone()
            self.compact_mask = visual_pos_masks.clone()
            self.compact_attention = attention_mask.clone()
            hidden = inputs_embeds.clone()
            hidden[visual_pos_masks] += deepstack_visual_embeds[0]
            return self.layers[0](hidden)

    language = Language()
    policy = SimpleNamespace(
        model=SimpleNamespace(
            qwen=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(language_model=language))
            )
        )
    )
    visual_token_pruning(policy, fraction, "uniform")
    x = torch.arange(36).float().reshape(1, 12, 3)
    mask = torch.tensor(
        [[False, True, True, True, True, False, True, True, True, True, False, False]]
    )
    positions = torch.arange(12).reshape(1, 1, 12).expand(4, 1, -1)
    deepstack = torch.arange(24).float().reshape(8, 3)
    result = language(
        inputs_embeds=x,
        visual_pos_masks=mask,
        position_ids=positions,
        attention_mask=torch.ones(1, 12),
        deepstack_visual_embeds=[deepstack],
    )
    kept = language.compact_positions[0, 0]
    assert result.shape == x.shape
    assert ((kept >= 1) & (kept <= 4)).any()
    assert ((kept >= 6) & (kept <= 9)).any()
    assert torch.equal(result[:, ~mask[0]], x[:, ~mask[0]])
    fused = x.clone()
    fused[mask] += deepstack
    assert torch.equal(result[:, kept], fused[:, kept])
    assert language.compact_attention.shape[-1] == len(kept)
    assert torch.equal(language.compact_mask, mask[:, kept])
