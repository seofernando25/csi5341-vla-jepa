"""Passive query-token lineage audit of pinned Qwen/pretrain/LIBERO artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from evaluation.common import ROOT, file_hash, read_json, write_json
from evaluation.models import artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier diagnostic results')
    cfg = read_json(artifact('baseline') / 'config.json')
    pretrain = read_json(artifact('pretrain') / 'config.json')
    for key in ('special_action_token', 'embodied_action_token', 'chunk_size', 'num_video_frames'):
        if cfg[key] != pretrain[key]:
            raise ValueError(f'Token configuration differs: {key}')
    tokenizer = AutoTokenizer.from_pretrained(artifact('qwen'), local_files_only=True)
    tokens = [cfg['special_action_token'].format(i) for i in range(cfg['chunk_size'] * 4)]
    tokens.append(cfg['embodied_action_token'])
    for token in tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    names, rows, keys = ('qwen', 'pretrain', 'baseline'), [], []
    for name in names:
        with safe_open(artifact(name) / 'model.safetensors', framework='pt', device='cpu') as handle:
            matched = [k for k in handle.keys() if k.endswith('language_model.embed_tokens.weight')]
            if len(matched) != 1:
                raise ValueError('Expected one Qwen input embedding matrix')
            keys.append(matched[0])
            rows.append(handle.get_tensor(matched[0])[ids].float())
    tubelet = read_json(artifact('world_model') / 'config.json')['tubelet_size']
    used = list(range(cfg['num_video_frames'] // tubelet - 1)) + [len(tokens) - 1]
    result = {'purpose': __doc__, 'diagnostic_source_sha256': file_hash(__file__),
        'sources': {k: read_json(ROOT / 'evaluation/sources.json')['models'][k] for k in names},
        'embedding_keys': keys, 'tokens': tokens, 'token_ids': ids, 'used_query_row_indices': used,
        'query_row_bytes_sha256': [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in rows],
        'transitions': [],
        'limitations': 'Embedding changes do not prove a causal control mechanism or the training corpus/recipe. '
        'Hidden width differs from SmolVLM. No weight modification.'}
    for a, b, label in [(0, 1, 'Qwen to Pretrain'), (1, 2, 'Pretrain to LIBERO'), (0, 2, 'Qwen to LIBERO')]:
        result['transitions'].append({'label': label,
            'per_row_delta_l2': torch.linalg.vector_norm(rows[b] - rows[a], dim=1).tolist()})
    write_json(args.output, result)


if __name__ == '__main__':
    main()
