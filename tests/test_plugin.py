import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.processor import TransitionKey

import lerobot_policy_vla_jepa_smolvlm  # noqa: F401
from lerobot_policy_vla_jepa_smolvlm.processor_vla_jepa_smolvlm import (
    DropImagePadMasksProcessorStep,
)
from lerobot_policy_vla_jepa_smolvlm.smolvlm_interface import ResidualRMSMLPAdapter, _LinearAdapter


def test_plugin_registration():
    assert "vla_jepa_smolvlm" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("vla_jepa_smolvlm").name == "vla_jepa_smolvlm"


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


def tiny_backbone():
    from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

    config = SmolVLMConfig(
        image_token_id=3,
        pad_token_id=0,
        scale_factor=2,
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
    )
    return SmolVLMForConditionalGeneration(config)


def test_smolvlm_real_decoder_projection_and_gradients(monkeypatch):
    from types import SimpleNamespace

    from torch import nn

    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig
    from lerobot_policy_vla_jepa_smolvlm import smolvlm_interface as interface
    from lerobot_policy_vla_jepa_smolvlm.modeling_vla_jepa_smolvlm import VLAJEPASmolVLMModel

    backbone = tiny_backbone()
    monkeypatch.setattr(
        interface.SmolVLMForConditionalGeneration, "from_pretrained", lambda *a, **k: backbone
    )
    monkeypatch.setattr(
        interface.AutoProcessor,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(tokenizer=SimpleNamespace()),
    )
    monkeypatch.setattr(
        interface.AutoImageProcessor, "from_pretrained", lambda *a, **k: SimpleNamespace()
    )
    cfg = VLAJEPASmolVLMConfig(torch_dtype="float32", conditioning_dim=32, init_from_vla_jepa=None)
    adapter = interface.SmolVLMInterface(cfg)
    model = VLAJEPASmolVLMModel.__new__(VLAJEPASmolVLMModel)
    nn.Module.__init__(model)
    model.qwen = adapter
    inputs = {
        "input_ids": torch.tensor([[1, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "pixel_values": torch.rand(1, 1, 3, 8, 8),
    }
    hidden = model._qwen_last_decoder_hidden(inputs)
    assert hidden.shape == (1, 4, 32)
    hidden.square().mean().backward()
    assert adapter.hidden_adapter.weight.grad is not None
    assert all(p.grad is None for p in backbone.parameters())
    assert not backbone.model.text_model.layers[-1]._forward_hooks
    cfg.unfreeze_last_n = 1
    cfg.train_multimodal_projector = True
    adapter._configure_trainability()
    assert all(p.requires_grad for p in backbone.model.text_model.layers[-1].parameters())
    assert not any(p.requires_grad for p in backbone.model.text_model.layers[0].parameters())
    assert all(p.requires_grad for p in backbone.model.connector.parameters())


def test_smolvlm_processor_receives_batched_images_without_rescaling():
    from types import SimpleNamespace

    from transformers import BatchFeature

    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig
    from lerobot_policy_vla_jepa_smolvlm.smolvlm_interface import SmolVLMInterface

    calls = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"][-1]["text"]

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return BatchFeature(
                {
                    "input_ids": torch.ones(2, 3, dtype=torch.long),
                    "pixel_values": torch.ones(2, 1, 3, 8, 8),
                }
            )

    adapter = SmolVLMInterface.__new__(SmolVLMInterface)
    torch.nn.Module.__init__(adapter)
    adapter.config = VLAJEPASmolVLMConfig(torch_dtype="bfloat16")
    adapter.model = SimpleNamespace(device=torch.device("cpu"))
    adapter.processor = Processor()
    images = [[torch.rand(3, 8, 8)], [torch.rand(3, 8, 8)]]
    inputs = adapter.build_inputs(images, ["pick", "place"], "<a>", "<e>")
    assert calls[0]["images_kwargs"]["do_rescale"] is False
    assert len(calls[0]["images"]) == 2
    assert "pick" in calls[0]["text"][0] and "place" in calls[0]["text"][1]
    assert inputs["input_ids"].dtype == torch.long
    assert inputs["pixel_values"].dtype == torch.bfloat16


def test_torchvision_input_path_never_copies_images_to_cpu(monkeypatch):
    from types import SimpleNamespace
    from transformers import BatchFeature
    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig
    from lerobot_policy_vla_jepa_smolvlm.smolvlm_interface import SmolVLMInterface

    calls=[]
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content'][-1]['text']
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return BatchFeature({'input_ids':torch.ones(1,3,dtype=torch.long),
                                 'pixel_values':torch.ones(1,1,3,8,8)})
    def forbidden_cpu(*args, **kwargs):
        raise AssertionError('Torchvision path copied an image to CPU')
    monkeypatch.setattr(torch.Tensor,'cpu',forbidden_cpu)
    adapter=SmolVLMInterface.__new__(SmolVLMInterface);torch.nn.Module.__init__(adapter)
    adapter.config=VLAJEPASmolVLMConfig(image_processor_backend='torchvision')
    adapter.model=SimpleNamespace(device=torch.device('cpu'));adapter.processor=Processor()
    image=torch.rand(3,8,8)
    out=adapter.build_inputs([[image]],['pick'],'<a>','<e>')
    assert calls[0]['images_kwargs']['device']==adapter.model.device
    assert calls[0]['images'][0][0].data_ptr()==image.data_ptr()
    assert out['input_ids'].dtype==torch.long


def test_real_smol_processor_preserves_unit_range_image_contrast():
    """The real kwargs merger must honor no-rescale, not just receive the flag."""
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models
    from transformers import (
        PreTrainedTokenizerFast, SmolVLMImageProcessor, SmolVLMProcessor, SmolVLMVideoProcessor,
    )
    from lerobot_policy_vla_jepa_smolvlm import VLAJEPASmolVLMConfig
    from lerobot_policy_vla_jepa_smolvlm.smolvlm_interface import SmolVLMInterface

    vocab = {"<unk>": 0, "<pad>": 1, "<image>": 2,
             "<fake_token_around_image>": 3, "<global-img>": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>", pad_token="<pad>", additional_special_tokens=list(vocab)[2:])
    processor = SmolVLMProcessor(
        SmolVLMImageProcessor(size={"longest_edge": 8}, max_image_size={"longest_edge": 8},
                             do_image_splitting=False),
        tokenizer, SmolVLMVideoProcessor(), image_seq_len=1,
        chat_template="{% for m in messages %}{% for c in m['content'] %}"
                      "{% if c['type'] == 'image' %}<image>{% else %}{{c['text']}}{% endif %}"
                      "{% endfor %}{% endfor %}")
    adapter = SmolVLMInterface.__new__(SmolVLMInterface)
    torch.nn.Module.__init__(adapter)
    adapter.config = VLAJEPASmolVLMConfig(torch_dtype="float32")
    adapter.model = SimpleNamespace(device=torch.device("cpu"))
    adapter.processor = processor
    image = torch.zeros(3, 8, 8)
    image[:, :, 4:] = 1
    pixels = adapter.build_inputs([[image]], ["pick"], "<a>", "<e>")["pixel_values"]
    assert torch.equal(pixels[0, 0], 2 * image - 1)
    # Demonstrate the previous call's failure using the actual dependency.
    legacy = processor(text=["<image>"], images=[[image]], return_tensors="pt",
                       do_rescale=False, images_kwargs={"device": "cpu"})["pixel_values"]
    assert legacy.max() - legacy.min() < .01
