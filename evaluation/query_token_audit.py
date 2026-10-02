"""Compare added query-token embeddings without loading or changing a policy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from evaluation.common import file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostics')
    configs = [read_json(p / 'config.json') for p in args.checkpoints]
    for key in ('special_action_token', 'embodied_action_token', 'chunk_size', 'vlm_model_name'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'Checkpoint token configuration differs: {key}')
    cfg = configs[0]
    tokenizer = AutoTokenizer.from_pretrained(artifact('smolvlm'), local_files_only=True)
    tokens = [cfg['special_action_token'].format(i) for i in range(cfg['chunk_size'] * 4)]
    tokens.append(cfg['embodied_action_token'])
    for token in tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    world_config = read_json(artifact('world_model') / 'config.json')
    tubelet = world_config['tubelet_size'] if cfg['enable_world_model'] else cfg['jepa_tubelet_size']
    prompt_steps = cfg['num_video_frames'] // tubelet - 1
    used_tokens = tokens[:prompt_steps] + [cfg['embodied_action_token']]
    rows, hashes = [], []
    for path in args.checkpoints:
        model = path / 'model.safetensors'
        hashes.append(file_hash(model))
        with safe_open(model, framework='pt', device='cpu') as handle:
            keys = [k for k in handle.keys() if k.endswith('text_model.embed_tokens.weight')]
            if len(keys) != 1:
                raise ValueError('Expected one SmolVLM input embedding matrix')
            key = keys[0]
            rows.append(handle.get_tensor(key)[ids].float())
    centered = rows[0] - rows[0].mean(0, keepdim=True)
    write_json(args.output, {
        'purpose': 'Passive query-token initialization and frozen-row audit',
        'checkpoint_sha256': hashes, 'diagnostic_source_sha256': file_hash(__file__),
        'embedding_key': key, 'tokens': tokens, 'token_ids': ids,
        'prompt_query_tokens': used_tokens,
        'prompt_query_repetitions': [cfg['num_action_tokens_per_timestep']] * prompt_steps
        + [cfg['num_embodied_action_tokens_per_instruction']],
        'query_rows_equal': bool(torch.equal(*rows)),
        'query_row_bytes_sha256': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in rows],
        'mean_row_l2': float(torch.linalg.vector_norm(rows[0], dim=1).mean()),
        'rms_deviation_from_row_mean': float(centered.square().mean().sqrt()),
        'maximum_coordinate_deviation_from_row_mean': float(centered.abs().max()),
        'limitations': 'Rows do not establish useful attention or a causal failure mechanism. '
        'Positional encoding and decoder layers can distinguish similar embeddings. No weight modification.'})


if __name__ == '__main__':
    main()
