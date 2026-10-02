"""Read-only localhost dashboard for the current Dream-RSI journal."""
import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web/rsi-dashboard'
_CACHE = {}


def read_record(path):
    stamp = path.stat().st_mtime_ns
    cached = _CACHE.get(path)
    if cached is None or cached[0] != stamp:
        cached = (stamp, json.loads(path.read_text()))
        _CACHE[path] = cached
    return cached[1]



def rollout_progress(path, record):
    """Read only completed CSV rows; tolerate an append in progress."""
    args = record.get('arguments') or {}
    tasks = args.get('tasks', list(range(10)))
    if not isinstance(tasks, list):
        tasks = list(range(10))
    total = int(args.get('episodes', 10)) * len(tasks)
    episodes = []
    if path.is_file():
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row.get('status') == 'completed' and row.get('success') in ('0', '1') and row.get('task_name'):
                    episodes.append(row)
    successes = 0
    curve = []
    for i, row in enumerate(episodes, 1):
        successes += int(row['success'])
        curve.append({'episode': i, 'success_rate': 100 * successes / i})
    return {'completed': len(episodes), 'total': total, 'successes': successes,
            'last_task': episodes[-1]['task_name'] if episodes else None, 'curve': curve}


def evaluation_history(root):
    """Keep paper evaluation and search evidence in distinct display datasets."""
    base = root / 'studies/evaluation'
    runs, benchmarks = [], []
    for path in sorted((base / 'runs').glob('*/run.json')):
        r = read_record(path)
        summary = r.get('summary', {})
        row = {'id': r['run_id'], 'variant': r.get('variant'), 'experiment': r.get('experiment'),
               'phase': r.get('phase'), 'status': r.get('status'), 'summary': summary,
               'arguments': r.get('arguments'), 'model': r.get('model'),
               'source': str(path.relative_to(root)), 'gpu': r.get('environment', {}).get('gpu')}
        if r.get('experiment') == 'rollout':
            row['progress'] = rollout_progress(path.parent / 'episodes.csv', r)
        runs.append(row)
        if r.get('status') == 'completed' and r.get('experiment') == 'benchmark':
            peaks = [p['allocated_bytes'] for p in summary.get('memory_peaks', [])]
            benchmarks.append({'id': row['id'], 'label': row['variant'], 'gpu': row['gpu'],
                               'memory': max(peaks)/1024**3 if peaks else None,
                               'latency': summary.get('pipeline', {}).get('median_ms'),
                               'predictions': summary.get('predictions_per_repetition'),
                               'repetitions': summary.get('repetitions'), 'detail': row})
    final = []
    path = base / 'analysis/summary.json'
    if path.is_file():
        report = read_record(path)
        if report.get('rollout_phase') == 'final':
            for variant, r in report.get('results', {}).items():
                t, s = r.get('timing', {}), r.get('success', {})
                final.append({'id': variant, 'label': variant, 'gpu': report.get('controls', {}).get('gpu'),
                              'success': 100*s['rate'] if number(s.get('rate')) is not None else None,
                              'ci95': [100*v for v in s.get('ci95', [])],
                              'latency': t.get('pipeline', {}).get('median_ms'),
                              'memory': t['peak_allocated_bytes']/1024**3 if t.get('peak_allocated_bytes') else None,
                              'detail': {'variant': variant, 'results': r, 'source': str(path.relative_to(root)),
                                         'evidence': report.get('evidence'), 'controls': report.get('controls')}})
    path = base / 'adaptation/summary.json'
    learning = []
    if path.is_file():
        for r in read_record(path).get('validation', []):
            learning.append({'id': f"S500 · {r['step']} steps", 'label': str(r['step']),
                             'steps': r['step'], 'loss': r['loss'],
                             'detail': {'id': f"S500 · {r['step']} steps", **r, 'source': str(path.relative_to(root))}})
    return {'runs': runs, 'benchmarks': benchmarks, 'final': final, 'learning': learning}


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def pareto(points, metric):
    valid = [p for p in points if number(p.get(metric)) is not None and number(p.get('loss')) is not None]
    return [p['id'] for p in valid if not any(
        q[metric] <= p[metric] and q['loss'] <= p['loss'] and
        (q[metric] < p[metric] or q['loss'] < p['loss']) for q in valid)]


def point(identifier, metrics, phase, baseline=False):
    return {'id': identifier, 'phase': phase, 'baseline': baseline,
            'loss': number(metrics.get('eval_loss')),
            'seconds': number(metrics.get('wall_seconds')),
            'memory': (number(metrics.get('hardware', {}).get('peak_allocated_bytes')) or 0) / 1024**3 or None,
            'gpu': metrics.get('hardware', {}).get('gpu', 'unknown'),
            'samples': metrics.get('eval_samples', {}).get('count'),
            'steps': metrics.get('optimizer_steps')}


def training_curve(path):
    stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    key = ('curve', path)
    if key in _CACHE and _CACHE[key][0] == stamp:
        return _CACHE[key][1]
    with path.open('rb') as handle:
        handle.seek(max(0, stamp[1] - 8 * 1024**2))
        text = handle.read().decode(errors='replace')
    step, total, points = None, None, {}
    # tqdm gives exact steps; LeRobot's INFO step field abbreviates thousands.
    for match in re.finditer(r'(?P<step>\d+)/(?P<total>\d+)\s*\[|(?P<line>INFO[^\r\n]*\bloss:[^\r\n]*)', text):
        if match.group('step'):
            step, total = int(match['step']), int(match['total'])
        elif step is not None:
            values = {k: float(v) for k, v in re.findall(r'\b(loss|action_loss|wm_loss|step_s|mem_gb):([\d.eE+-]+)', match['line'])}
            if number(values.get('loss')) is not None:
                points[step] = {'step': step, **values}
    result = {'step': step, 'total': total, 'curve': list(points.values())}
    _CACHE[key] = (stamp, result)
    return result


def recovery_live(root):
    """Show the amended recovery study separately from legacy confirmation."""
    folder = root / 'outputs/recovery/cloud'
    path = folder / 'remote-job.json'
    job = read_record(path) if path.is_file() else {}
    setup_path = folder / 'setup-status.json'
    setup = read_record(setup_path) if setup_path.is_file() else {}
    if not job and not setup:
        return None
    if not job:
        return {'arm': 'RGB-n0008', 'stage': setup.get('phase', 'preparing').replace('_', ' '),
                'status': 'running', 'cloud': True,
                'progress': {'label': 'RGB-n0008', 'phase': 'recovery setup', 'step': 0,
                             'total': None, 'curve': [], 'completed': False,
                             'updated_at': setup_path.stat().st_mtime, 'cloud': True}}
    points = {}
    updated = path.stat().st_mtime
    for record in sorted((root / 'studies/recovery/training').glob('*/run.json')):
        data = read_record(record)
        if (data.get('study') != job.get('study')
                or data.get('recipe_sha256') != job.get('recipe_sha256')
                or data.get('purpose') != 'recovery_adaptation'):
            continue
        metrics = record.parent / 'metrics.jsonl'
        if not metrics.is_file():
            continue
        updated = max(updated, metrics.stat().st_mtime)
        with metrics.open('rb') as handle:
            handle.seek(max(0, metrics.stat().st_size - 8 * 1024**2))
            for line in handle.read().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('phase') == 'training' and number(row.get('loss')) is not None:
                    points[row['step']] = {'step': row['step'], **{k: row.get(k) for k in ['loss', 'action_loss', 'wm_loss']}}
    curve = [points[k] for k in sorted(points)]
    visible = curve[::max(1, len(curve)//300)]
    if curve and (not visible or visible[-1] != curve[-1]):
        visible.append(curve[-1])
    done = job.get('status') in {'completed', 'failed'}
    label = 'Query-n0008' if job.get('study', '').startswith('n0008-query-') else 'RGB-n0008'
    return {'arm': label, 'stage': 'RTX 5090 · ' + job.get('status', 'preparing'),
            'status': job['status'] if done else 'running', 'cloud': True,
            'progress': {'label': label, 'phase': 'recovery · ' + job.get('status', 'preparing'),
                         'step': curve[-1]['step'] if curve else 0, 'total': job.get('target_step'),
                         'curve': visible, 'completed': job.get('status') == 'completed',
                         'updated_at': updated, 'cloud': True}}


def local_rate_jobs(root):
    """Registered local branches, with one mean loss per accumulated optimizer update."""
    path = root / 'outputs/recovery/local-lr-pair/job.json'
    registration_path = root / 'studies/recovery/local_lr_pair_registration.json'
    if not path.is_file() or not registration_path.is_file():
        return []
    job, registration = read_record(path), read_record(registration_path)
    if (job.get('study') != registration['study'] or job.get('registration_sha256') !=
            hashlib.sha256(registration_path.read_bytes()).hexdigest()):
        return []
    result = []
    for branch in registration['execution_order']:
        points, updated = {}, path.stat().st_mtime
        records = []
        for record_path in sorted((root / 'studies/recovery/training').glob('*/run.json')):
            record = read_record(record_path)
            if (record.get('recipe_sha256') != registration['recipe_sha256'][branch]
                    or record.get('source_manifest') != registration['source_manifest']
                    or record.get('purpose') != 'recovery_adaptation'):
                continue
            metrics = record_path.parent / 'metrics.jsonl'
            if metrics.is_file():
                rows = []
                for line in metrics.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get('phase') == 'training' and number(row.get('loss')) is not None:
                        step = row.get('step')
                        if isinstance(step, int) and step > 0 and row.get('optimizer_updates', step // 2) == step // 2:
                            rows.append(row)
                records.append((record_path.parent, metrics, rows))
        for _, metrics, rows in records:
            if rows:
                # A resumed stream supersedes an earlier uncommitted tail.
                start = min(row['step'] for row in rows)
                points = {step: row for step, row in points.items() if step < start}
                points.update({row['step']: row for row in rows})
            updated = max(updated, metrics.stat().st_mtime)
        active = job.get('active_branch') == branch
        if active and records and not records[-1][2]:
            points = {step: row for step, row in points.items() if step <= job.get('resume_microstep', 0)}
        curve = []
        for step in sorted(points):
            if step % 2 or step - 1 not in points:
                continue
            pair = [points[step - 1], points[step]]
            averaged = {'step': step // 2}
            for key in ('loss', 'action_loss', 'wm_loss'):
                values = [row.get(key) for row in pair]
                averaged[key] = sum(values) / 2 if all(number(value) is not None for value in values) else None
            curve.append(averaged)
        visible = curve[::max(1, len(curve) // 300)]
        if curve and visible[-1] != curve[-1]:
            visible.append(curve[-1])
        live = False
        pid = job.get('child_pid')
        if active and job.get('status') == 'training' and isinstance(pid, int) and pid > 1:
            try:
                command = Path(f'/proc/{pid}/cmdline').read_bytes()
                live = b'evaluation.recovery_train' in command and f'local_lr_{branch}_config.json'.encode() in command
            except OSError:
                pass
        complete = branch in job.get('branches', {})
        status = 'completed' if complete else 'running' if live else 'last observed' if points else 'queued'
        result.append({'id': f'local-rate-{branch}', 'status': status,
                       'label': f'Local RTX3090 · {branch} rate',
                       'progress': {'label': f'Local {branch} rate', 'phase': 'learning-rate comparison',
                                    'step': max(points, default=0) // 2, 'total': 500,
                                    'native_microstep': max(points, default=0),
                                    'native_total_microsteps': 1000, 'accumulation_steps': 2,
                                    'curve': visible, 'completed': complete, 'updated_at': updated,
                                    'cloud': False, 'unit': 'optimizer updates', 'averaged_microbatches': 2}})
    return result


def confirmation_live(root):
    recovery = recovery_live(root)
    if recovery:
        return recovery
    path = root / 'outputs/confirmation/n0008-20261001/status.json'
    cloud_path = root / 'outputs/cloud/remote-job.json'
    cloud = read_record(cloud_path) if cloud_path.is_file() else {}
    if cloud.get('status') in {'preparing', 'timing', 'training', 'trained', 'evaluating', 'completed', 'failed', 'interrupted'}:
        path = cloud_path
        status = {
            'arm': cloud.get('preflight_label', 'RSI-n0008-5090-preflight') if cloud['status'] == 'timing'
                   else cloud.get('training_label', 'RSI-n0008-5090'),
            'stage': 'RTX 5090 · ' + cloud['status'],
            'status': cloud['status'] if cloud['status'] in {'completed', 'failed', 'interrupted'} else 'running',
            'cloud': True,
        }
    elif path.is_file():
        status = read_record(path)
    else:
        return None
    label = status.get('arm')
    if label not in {'RSI-n0008', 'SmolVLM-GPU-base'} and not re.fullmatch(r'RSI-n0008-5090(?:-r\d+)?(?:-preflight)?', label or ''):
        return status
    runs = []
    for record in sorted((root / 'studies/evaluation/training').glob('*/run.json')):
        data = read_record(record)
        if data.get('variant') == label:
            runs.append((record, data))
    points = {}
    updated = path.stat().st_mtime
    for record, data in runs:
        metrics = record.parent / 'metrics.jsonl'
        if not metrics.is_file():
            continue
        updated = max(updated, metrics.stat().st_mtime)
        with metrics.open('rb') as handle:
            handle.seek(max(0, metrics.stat().st_size - 8 * 1024**2))
            for line in handle.read().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('phase') == 'training' and number(row.get('loss')) is not None:
                    points[row['step']] = {'step': row['step'], **{k: row.get(k) for k in ['loss','action_loss','wm_loss']}}
    latest = runs[-1][1] if runs else {}
    curve = [points[k] for k in sorted(points)]
    # Thin display points; retain the most recent measurement exactly.
    visible = curve[::max(1, len(curve)//300)]
    if curve and (not visible or visible[-1] != curve[-1]):
        visible.append(curve[-1])
    status = dict(status)
    status['progress'] = {'label': label, 'phase': 'confirmation · '+status.get('stage',''),
                          'step': curve[-1]['step'] if curve else 0,
                          'total': latest.get('requested_steps'), 'curve': visible,
                          'completed': status.get('status') == 'completed', 'updated_at': updated,
                          'cloud': bool(status.get('cloud'))}
    return status


def proposal_history(state, identifier):
    """Read public authored messages only; exclude tool output and reasoning."""
    if not re.fullmatch(r'n\d+', identifier):
        return []
    history = []
    for path in sorted((state / 'attempts' / identifier).glob('proposal_*/discovery/stdout.log')):
        messages, proposal = [], None
        with path.open('rb') as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - 262144))
            for line in handle.read().decode(errors='replace').splitlines():
                try:
                    item = json.loads(line).get('item', {})
                except (ValueError, AttributeError):
                    continue
                if item.get('type') != 'agent_message' or not isinstance(item.get('text'), str):
                    continue
                message = item['text'][:24000]
                try:
                    draft = json.loads(message)
                except ValueError:
                    messages.append(message)
                    continue
                if isinstance(draft, dict) and isinstance(draft.get('hypothesis'), str) and isinstance(draft.get('structural_change'), str):
                    proposal = draft
        if messages or proposal:
            history.append({'retry': path.parents[1].name, 'source': str(path.relative_to(state)),
                            'messages': messages[-8:], 'proposal': proposal})
    return history


def snapshot(root=ROOT, service=True, state_name='.rsi'):
    allowed = {'.rsi', *(p.name for p in root.glob('.rsi-archive-pil-*') if p.is_dir())}
    if state_name not in allowed:
        raise ValueError('Unknown study')
    state = root / state_name
    events, warnings = [], []
    for path in sorted((state / 'events').glob('*.json')):
        try:
            events.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            warnings.append('A journal record is temporarily unreadable; refresh will retry.')
    config = next((e['config'] for e in events if e['kind'] == 'initialized'), {})
    batches, attempts, points = {}, {}, []
    for e in events:
        kind = e['kind']
        identifier = e.get('attempt')
        if kind == 'batch_started':
            batches[e['batch_id']] = {'id': e['batch_id'], 'cycle': e['outer'] + 1,
                                      'at': e.get('at'), 'parents': e.get('planned_actions', [])}
        if kind == 'attempt':
            attempts[identifier] = {'id': identifier, 'batch': e.get('batch_id'),
                                    'parent': e.get('parent'), 'cycle': e.get('outer', 0) + 1,
                                    'status': 'proposing', 'at': e.get('started_at'),
                                    'proposal': None, 'rejections': [], 'loss': None, 'timeline': []}
        if identifier in attempts:
            row = attempts[identifier]
            row['timeline'].append({k: v for k, v in e.items() if k not in {'node', 'config', 'workspace_hash'}})
            if kind == 'proposal_rejected':
                row['rejections'].append({'retry': e.get('retry'), 'reason': e.get('reason'),
                                           'mechanism': e.get('mechanism')})
            if kind == 'proposal_accepted':
                row.update(proposal=e.get('proposal'), status='queued for evaluation', family=e.get('family_id'))
            if kind == 'outcome':
                node = e['node']
                row.update(status=node['status'], summary=node.get('summary'),
                           proposal=node.get('proposal') or row['proposal'], ended_at=e.get('ended_at'),
                           metrics=node.get('metrics', {}))
                if node['status'] == 'ok':
                    p = point(identifier, node.get('metrics', {}), 'screen')
                    points.append(p); row['loss'] = p['loss']; row['seconds'] = p['seconds']
            if kind == 'promotion':
                row['promotion'] = {'status': e['status'], 'delta': e.get('delta_vs_baseline')}
                if e['status'] == 'ok':
                    points.append(point(identifier + ' · promotion', e['metrics'], 'promotion'))
        if kind in {'baseline', 'promotion_baseline'}:
            points.append(point('baseline' if kind == 'baseline' else 'promotion baseline', e['metrics'],
                                'screen' if kind == 'baseline' else 'promotion', True))
    for row in attempts.values():
        row['proposal_history'] = proposal_history(state, row['id'])
        drafts = [h['proposal'] for h in row['proposal_history'] if h['proposal']]
        row['draft_proposal'] = drafts[-1] if drafts else None
    # Observe log tails only; never lock or write the live journal.
    logs = list(state.glob('baseline/*/probe/stderr.log')) + list(state.glob('attempts/*/probe/stderr.log'))
    logs += list(state.glob('promotion/**/probe/stderr.log'))
    logs += list(state.glob('attempts/*/promotion/probe/stderr.log'))
    logs = [p for p in logs if p.is_file()]
    progress = None
    if logs:
        latest = max(logs, key=lambda p: p.stat().st_mtime)
        progress = {'log': str(latest.relative_to(state)), 'updated_at': latest.stat().st_mtime,
                    **training_curve(latest)}
        parts = progress['log'].split('/')
        promotion = parts[0] == 'promotion' or 'promotion' in parts[2:]
        identifier = parts[1] if parts[0] == 'attempts' else None
        label = identifier or 'Baseline'
        phase = 'promotion' if promotion else 'screen'
        done = any((e['kind'] == 'promotion' if promotion else e['kind'] == 'outcome') and
                   e.get('attempt') == identifier for e in events) if identifier else any(
                       e['kind'] == ('promotion_baseline' if promotion else 'baseline') for e in events)
        progress.update(label=label, phase=phase, completed=done)
    active = 'unknown'
    if state_name != '.rsi':
        service = False
        active = 'archived'
    if service:
        try:
            active = subprocess.run(['systemctl', '--user', 'is-active', 'csi5341-rsi-overnight'],
                                    capture_output=True, text=True, timeout=2).stdout.strip() or 'unknown'
        except (OSError, subprocess.TimeoutExpired):
            pass
    stage = 'Search running' if any(e['kind'] == 'baseline' for e in events) else 'Measuring the clean baseline'
    if active == 'active':
        if progress and not progress['completed']:
            stage = f"{progress['label']} · {progress['phase']}"
        for row in attempts.values():
            if row['status'] == 'proposing':
                row['active'] = True
                row['status'] = 'building proposal'
        if progress and progress['log'].startswith('attempts/'):
            identifier = progress['log'].split('/')[1]
            row = attempts.get(identifier)
            if row and (row['status'] == 'queued for evaluation' or
                        ('/promotion/' in progress['log'] and not row.get('promotion'))):
                row.update(active=True, status='evaluating promotion' if '/promotion/' in progress['log'] else 'evaluating screen')
                stage = f"Evaluating {identifier}"
    return {'updated_at': datetime.now(timezone.utc).isoformat(), 'service': active, 'stage': stage,
            'study': state_name, 'studies': [{'id': name, 'label': 'Active search · GPU processor' if name == '.rsi' else 'Previous search · PIL processor'} for name in sorted(allowed, key=lambda x:x!='.rsi')],
            'initialized': bool(config), 'config': {'screen_steps': config.get('probe_steps'),
            'promotion_steps': config.get('promotion_steps'), 'max_cycles': config.get('max_outer_iterations')},
            'counts': {'batches': len(batches), 'attempts': len(attempts),
                       'measured': sum(number(r.get('loss')) is not None for r in attempts.values()),
                       'failed': sum(r['status'] in {'runtime_failure', 'implementation_failure', 'proposal_failure', 'interrupted'} for r in attempts.values())},
            'batches': list(batches.values()), 'attempts': list(attempts.values()), 'points': points,
            'progress': progress, 'disk_free_gib': shutil.disk_usage(root).free / 1024**3,
            'evaluation': evaluation_history(root),
            'confirmation': confirmation_live(root) if state_name == '.rsi' else None,
            'local_training_jobs': local_rate_jobs(root) if state_name == '.rsi' else [],
            'events': [{'kind': e['kind'], 'at': e.get('at') or e.get('ended_at') or e.get('started_at'),
                        'attempt': e.get('attempt'), 'batch': e.get('batch_id'), 'reason': e.get('reason')}
                       for e in events[-60:]], 'warnings': warnings}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?', 1)[0]
        try:
            if path == '/api/state':
                study = parse_qs(urlsplit(self.path).query).get('study', ['.rsi'])[0]
                payload, mime = json.dumps(snapshot(state_name=study), allow_nan=False).encode(), 'application/json'
            elif path in {'/', '/app.js', '/style.css'}:
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[path]
                payload = (WEB / name).read_bytes()
                mime = {'/': 'text/html', '/app.js': 'text/javascript', '/style.css': 'text/css'}[path]
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', mime + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        except (OSError, ValueError):
            self.send_error(503, 'Study snapshot unavailable; retry shortly')

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Dream-RSI dashboard: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
