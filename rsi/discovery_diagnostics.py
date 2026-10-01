"""Staged trusted exporter: current-study observations for discovery, no live mounts."""
import hashlib
import json
import math
from pathlib import Path
import re

FIELDS = ('loss', 'action_loss', 'wm_loss', 'grdn', 'lr', 'data_s', 'prep_s', 'updt_s', 'step_s', 'mem_gb')


def logged_samples(text):
    step, total, rows = None, None, {}
    for m in re.finditer(r'(?P<step>\d+)/(?P<total>\d+)\s*\[|(?P<line>INFO[^\r\n]*\bloss:[^\r\n]*)', text):
        if m.group('step'):
            step, total = int(m['step']), int(m['total'])
        elif step is not None:
            values = {k: float(v) for k, v in re.findall(r'\b([a-z_]+):([\d.eE+-]+)', m['line']) if k in FIELDS and math.isfinite(float(v))}
            if 'loss' in values:
                rows[step] = {'step': step, 'total': total, **values}
    return list(rows.values())


def prepare_evidence(state, events, destination):
    """Events must be the completed prefix frozen before proposing a batch."""
    state, destination = Path(state), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for e in events:
        kind = e['kind']
        if kind not in {'baseline','promotion_baseline','outcome','promotion'}:
            continue
        identifier = e.get('attempt')
        if identifier and not re.fullmatch(r'n\d+',identifier):
            continue
        node = e.get('node', {})
        metrics = e.get('metrics',node.get('metrics',{}))
        phase = 'promotion' if kind in {'promotion','promotion_baseline'} else 'screen'
        label = identifier or 'baseline'
        if identifier:
            logs = [state/'attempts'/identifier/('promotion/probe/stderr.log' if phase=='promotion' else 'probe/stderr.log')]
        else:
            pattern = 'promotion/baseline/*/probe/stderr.log' if phase=='promotion' else 'baseline/*/probe/stderr.log'
            logs = sorted(state.glob(pattern))
        run = {'id':label,'phase':phase,'status':e.get('status',node.get('status','ok')),
               'event_seq':e.get('seq'),'metrics':{k:metrics[k] for k in ('eval_loss','score','optimizer_steps','wall_seconds','attempt_wall_seconds','metric_provenance','eval_samples') if k in metrics},
               'hardware':{k:metrics.get('hardware',{}).get(k) for k in ('gpu','peak_allocated_bytes')},
               'inference_latency_ms':None,'logs':[]}
        for i,log in enumerate(logs):
            if not log.is_file() or log.is_symlink():
                continue
            with log.open('rb') as h:
                h.seek(max(0,log.stat().st_size-8*1024**2));raw=h.read()
            samples=logged_samples(raw.decode(errors='replace'))
            name=f'{label}-{phase}-{i}.jsonl'
            (destination/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in samples))
            run['logs'].append({'file':name,'source':str(log.relative_to(state)), 'source_tail_sha256':hashlib.sha256(raw).hexdigest(),'samples':len(samples)})
        run['parent'] = e.get('parent', node.get('parent'))
        run['proposal'] = node.get('proposal') or next((r.get('proposal') for r in events if r['kind']=='proposal_accepted' and r.get('attempt')==identifier), None)
        run['failure_reason'] = str(node.get('summary', e.get('summary', '')))[:800] if run['status'] != 'ok' else None
        groups = {}
        for parameter in metrics.get('trainability', []):
            group = '.'.join(parameter.get('name','unknown').split('.')[:2])
            groups[group] = groups.get(group, 0) + parameter.get('parameters', 0)
        run['trainable_parameters_by_component'] = groups
        records.append(run)
    (destination/'index.json').write_text(json.dumps({'runs':records,'interpretation':'Training loss is not held-out loss or LIBERO success. step_s is training step time, not inference latency. Missing measurements are unknown. Evidence is a read-only snapshot from this study before the current batch.'},indent=2,allow_nan=False))
    return records
