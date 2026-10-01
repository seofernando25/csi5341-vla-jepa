"""Prune completed evaluation checkpoints after preserving compact provenance.

Keeps milestone inference weights and the selected checkpoint's optimizer state.
Never touches datasets, model caches, the active RSI journal, or candidate source.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    queue = json.loads((ROOT / 'outputs/evaluation/final_queue.json').read_text())
    if queue['status'] != 'completed':
        raise RuntimeError('Final evaluation must be completed before cleanup')
    selected = json.loads((ROOT / 'studies/evaluation/selection.json').read_text())
    selected_dir = (ROOT / selected['checkpoint']['artifact']).parent
    keep_steps = {1500, 5000, selected['selected_step']}
    records = {}
    for path in (ROOT / 'studies/evaluation/training').glob('*/run.json'):
        run = json.loads(path.read_text())
        if run.get('status') == 'completed':
            for checkpoint in run.get('checkpoints', []):
                records[checkpoint['artifact']] = checkpoint
    removals = []
    for directory in selected_dir.parent.iterdir():
        if directory.is_symlink() or not directory.is_dir() or not directory.name.isdigit():
            continue
        artifact = str((directory / 'pretrained_model').relative_to(ROOT))
        evidence = records.get(artifact)
        if not evidence:
            continue
        weights = directory / 'pretrained_model/model.safetensors'
        if not weights.is_file() or weights.stat().st_size != evidence['bytes']:
            raise RuntimeError(f'Checkpoint differs from recorded evidence: {artifact}')
        targets = [directory] if int(directory.name) not in keep_steps else (
            [directory / 'training_state'] if directory != selected_dir else [])
        for target in targets:
            if not target.is_dir() or target.is_symlink():
                continue
            size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink())
            removals.append({'path': str(target.relative_to(ROOT)), 'bytes': size,
                             'checkpoint_evidence': evidence})
    print(json.dumps({'remove_count': len(removals), 'reclaim_GiB': sum(r['bytes'] for r in removals)/1024**3}))
    if args.apply and removals:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        audit = ROOT / 'studies/evaluation/retention' / f'{stamp}.json'
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({'reason': 'user-requested compact retention',
                                    'selected_checkpoint': selected['checkpoint'],
                                    'preserved_milestone_steps': sorted(keep_steps),
                                    'removed': removals}, indent=2) + '\n')
        for removal in removals:
            shutil.rmtree(ROOT / removal['path'])


if __name__ == '__main__':
    main()
