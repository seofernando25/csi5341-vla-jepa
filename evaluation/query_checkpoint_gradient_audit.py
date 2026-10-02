"""One registered backward comparison; never constructs or steps an optimizer."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

from evaluation.common import ROOT, environment, file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous audit measurements')
    registration_path = ROOT / 'studies/recovery/query_checkpoint_gradient_registration.json'
    registration = read_json(registration_path)
    source = args.architecture_source.resolve()
    manifest = {str(p.relative_to(source)): file_hash(p) for p in sorted((source / 'src').rglob('*.py'))}
    if manifest != registration['source_manifest'] or file_hash(args.checkpoint / 'model.safetensors') != registration['checkpoint_sha256']:
        raise ValueError('Source or weights differ from the registered query500')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_name() != registration['gpu']:
        raise ValueError('Wrong GPU for the registered diagnostic')
    sys.path.insert(0, str(source / 'src'))
    import lerobot_policy_vla_jepa_smolvlm as plugin
    if not Path(plugin.__file__).resolve().is_relative_to(source / 'src'):
        raise ValueError('Wrong plugin imported')
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.unfreeze_last_n != 32 or cfg.query_token_adaptation != 'input_residual':
        raise ValueError('Use the actual full-decoder query checkpoint')
    cfg.vlm_model_name, cfg.jepa_encoder_name = str(artifact('smolvlm')), str(artifact('world_model'))
    cfg.init_from_vla_jepa, cfg.device = None, 'cuda'
    policy = plugin.VLAJEPASmolVLMPolicy(cfg).to('cuda')
    policy.model.video_encoder.requires_grad_(False)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    load_model(policy, args.checkpoint / 'model.safetensors', strict=True, device='cpu')
    with safe_open(args.checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
        for key, value in policy.state_dict().items():
            if not torch.equal(value.detach().cpu(), saved.get_tensor(key)):
                raise ValueError('Checkpoint values or precision changed')
    metadata = LeRobotDatasetMetadata('local/libero_spatial', root=args.dataset_root)
    rename = {'observation.images.wrist_image': 'observation.images.image2'}
    episode, frame, seed = [registration[key] for key in ('episode', 'frame', 'seed')]
    if episode not in read_json(ROOT / 'studies/evaluation/training_split.json')['train_episodes']:
        raise ValueError('Diagnostic frame is not in the registered training split')
    dataset = LeRobotDataset('local/libero_spatial', root=args.dataset_root, episodes=[episode],
        delta_timestamps=resolve_delta_timestamps(cfg, metadata, rename), revision='v3.0',
        video_backend='pyav', return_uint8=True)
    sample = dataset[[int(x) for x in dataset.hf_dataset['frame_index']].index(frame)]
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'rename_observations_processor': {'rename_map': {}}})
    batch = _preprocess_dataset_batch(torch.utils.data.default_collate([sample]), metadata.camera_keys, rename, pre)
    result = {'purpose': registration['purpose'], 'status': 'running', 'environment': environment(),
              'registration_sha256': file_hash(registration_path), 'diagnostic_sha256': file_hash(__file__),
              'source_manifest': manifest, 'checkpoint_sha256': registration['checkpoint_sha256'],
              'episode': episode, 'frame': frame, 'seed': seed, 'modes': [],
              'limitations': registration['limitations']}
    write_json(args.output, result)
    references = {}
    policy.train()
    try:
        for mode in registration['modes']:
            enabled = mode == 'enabled_nonreentrant'
            if enabled:
                policy.model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            else:
                policy.model.qwen.model.gradient_checkpointing_disable()
            if policy.model.qwen.model.is_gradient_checkpointing != enabled:
                raise ValueError('Requested activation checkpointing state did not apply')
            policy.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(seed)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = policy.model(**policy._prepare_model_inputs(batch, training=True))
                total = losses['action_loss'] + losses['wm_loss']
            total.backward()
            fingerprint, squared_norm, squared_difference = hashlib.sha256(), 0.0, 0.0
            exact, mismatches, maximum_difference, names, elements = True, [], 0.0, [], 0
            for name, parameter in policy.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().cpu().contiguous()
                if not torch.isfinite(gradient).all():
                    raise ValueError('Nonfinite native gradient')
                names.append(name)
                elements += gradient.numel()
                fingerprint.update(name.encode())
                fingerprint.update(gradient.numpy().tobytes())
                squared_norm += float(gradient.square().sum())
                if mode == 'disabled':
                    references[name] = gradient.clone()
                else:
                    if name not in references or gradient.shape != references[name].shape:
                        raise ValueError('Populated gradient membership changed')
                    equal = torch.equal(gradient, references[name])
                    exact = exact and equal
                    if not equal:
                        difference = gradient - references[name]
                        squared_difference += float(difference.square().sum())
                        maximum_difference = max(maximum_difference, float(difference.abs().max()))
                        mismatches.append(name)
            if not names or (mode != 'disabled' and set(names) != set(references)):
                raise ValueError('Populated gradient membership changed')
            row = {'mode': mode, 'checkpointing_enabled': enabled,
                   'losses': {key: float(value.detach()) for key, value in losses.items()},
                   'gradient_sha256': fingerprint.hexdigest(), 'gradient_parameters': len(names),
                   'gradient_elements': elements, 'gradient_l2': math.sqrt(squared_norm),
                   'all_gradients_exactly_equal_to_disabled': exact, 'different_parameter_gradients': len(mismatches),
                   'first_differing_parameters': mismatches[:10],
                   'gradient_difference_l2': math.sqrt(squared_difference),
                   'maximum_gradient_difference': maximum_difference,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            result['modes'].append(row)
            write_json(args.output, result)
            print(row, flush=True)
            del losses, total
        result['status'] = 'completed'
        # Do not assert equality: this diagnostic must report a genuine difference.
        result['restored_losses_exact'] = result['modes'][0]['losses'] == result['modes'][2]['losses']
        result['restored_gradients_exact'] = result['modes'][2]['all_gradients_exactly_equal_to_disabled']
        with safe_open(args.checkpoint / 'model.safetensors', framework='pt', device='cpu') as saved:
            for key, value in policy.state_dict().items():
                if not torch.equal(value.detach().cpu(), saved.get_tensor(key)):
                    raise ValueError('Diagnostic altered saved parameter values')
        result['saved_weights_exact_after_all_backwards'] = True
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(args.output, result)


if __name__ == '__main__':
    main()
