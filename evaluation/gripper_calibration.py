"""Passive threshold audit of saved SmolVLM predictions; changes no processor.

Exclude whole padded chunks rather than infer masks absent from older probes.
Default-threshold classifications must reproduce the recorded native commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.common import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous calibration diagnostics')
    summary = read_json(args.audit / 'summary.json')
    probes = read_json(args.audit / 'probes.json')
    measured = list(csv.DictReader((args.audit / 'actions.csv').open()))
    thresholds = [-.75, -.5, -.25, 0., .25, .5, .75]
    selected, excluded = [], []
    for row, probe in zip(measured, probes, strict=True):
        if (row['split'], int(row['episode']), int(row['frame'])) != (
                probe['split'], probe['episode'], probe['frame']):
            raise ValueError('Prediction and measured frame membership differ')
        prediction = np.asarray(probe['predicted_normalized'])[0, :, 6].clip(-1, 1)
        normalized_target = np.asarray(probe['target_normalized'])[0, :, 6]
        expected_open = np.asarray(probe['target_physical'])[0, :, 6] < 0
        actual_open = np.asarray(probe['predicted_physical'])[0, :, 6] < 0
        if not np.array_equal(prediction >= .5, actual_open):
            raise ValueError('Default calibration does not reproduce native postprocessing')
        if not np.allclose(np.abs(normalized_target), 1) or not np.array_equal(normalized_target >= 0, expected_open):
            raise ValueError('Expected the recorded Smol -1/+1 gripper label space')
        identity = {'split': row['split'], 'episode': probe['episode'], 'frame': probe['frame']}
        if int(row['valid_actions']) != len(prediction):
            excluded.append(identity)
            continue
        selected.append((identity, prediction, expected_open))
    result = {'checkpoint_sha256': summary['checkpoint_sha256'],
              'source_manifest': summary['source_manifest'],
              'probes_sha256': file_hash(args.audit / 'probes.json'),
              'excluded_padded_chunks': excluded,
              'native_threshold': .5, 'splits': {},
              'limitations': 'Passive CPU calculation on saved diagnostic frames, not calibration selection, a rollout, or a change to the registered processor. Any deployed threshold change requires a separate registered inference variant.'}
    for split in ['train', 'heldout']:
        rows = [(identity, p, t) for identity, p, t in selected if identity['split'] == split]
        p = np.concatenate([p for _, p, _ in rows])
        t = np.concatenate([t for _, _, t in rows])
        result['splits'][split] = {'frames': len(rows), 'actions': len(p),
            'membership': [identity for identity, _, _ in rows],
            'thresholds': [{'threshold': threshold, 'errors': int(np.count_nonzero((p >= threshold) != t)),
                            'error_fraction': float(np.mean((p >= threshold) != t))} for threshold in thresholds]}
    write_json(args.output, result)
    print(json.dumps({split: rows['thresholds'] for split, rows in result['splits'].items()}))


if __name__ == '__main__':
    main()
