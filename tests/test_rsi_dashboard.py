import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).parents[1] / 'scripts/rsi_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def test_pareto_minimizes_both_metrics_and_keeps_ties():
    points = [{'id': 'a', 'loss': .4, 'memory': 4}, {'id': 'b', 'loss': .3, 'memory': 5},
              {'id': 'dominated', 'loss': .5, 'memory': 5}, {'id': 'tie', 'loss': .4, 'memory': 4},
              {'id': 'missing', 'loss': None, 'memory': 1}]
    assert dashboard.pareto(points, 'memory') == ['a', 'b', 'tie']


def test_snapshot_preserves_history_and_excludes_failed_metrics(tmp_path):
    journal = tmp_path / '.rsi/events'; journal.mkdir(parents=True)
    events = [dict(kind='initialized', config={'probe_steps': 500}),
              dict(kind='batch_started', batch_id='b1', outer=0, planned_actions=['root']),
              dict(kind='attempt', attempt='n0001', batch_id='b1', outer=0, parent='root'),
              dict(kind='proposal_accepted', attempt='n0001', proposal={'structural_change': 'adapter'}),
              dict(kind='outcome', attempt='n0001', node={'status':'runtime_failure', 'metrics':{'eval_loss':0}})]
    for i,event in enumerate(events):
        (journal / f'{i:08}.json').write_text(json.dumps(event))
    before = {p.name:p.read_bytes() for p in journal.iterdir()}
    result = dashboard.snapshot(tmp_path, service=False)
    assert result['counts']['batches'] == 1
    assert result['attempts'][0]['proposal']['structural_change'] == 'adapter'
    assert result['points'] == []
    assert before == {p.name:p.read_bytes() for p in journal.iterdir()}


def test_promotion_has_separate_protocol(tmp_path):
    journal=tmp_path/'.rsi/events'; journal.mkdir(parents=True)
    events=[{'kind':'attempt','attempt':'n1','outer':0},
            {'kind':'outcome','attempt':'n1','node':{'status':'ok','metrics':{'eval_loss':.4,'wall_seconds':10}}},
            {'kind':'promotion','attempt':'n1','status':'ok','metrics':{'eval_loss':.2,'wall_seconds':30}}]
    for i,e in enumerate(events): (journal/f'{i:08}.json').write_text(json.dumps(e))
    assert [p['phase'] for p in dashboard.snapshot(tmp_path,False)['points']] == ['screen','promotion']


def test_earlier_evaluations_remain_separate_from_search(tmp_path):
    base = tmp_path / 'studies/evaluation'
    run = base / 'runs/benchmark'; run.mkdir(parents=True)
    record = {'run_id':'benchmark', 'variant':'B16', 'experiment':'benchmark',
              'phase':'development', 'status':'completed', 'environment':{'gpu':'gpu'},
              'summary':{'pipeline':{'median_ms':10}, 'memory_peaks':[{'allocated_bytes':2**30}],
                         'predictions_per_repetition':500, 'repetitions':3}}
    (run/'run.json').write_text(json.dumps(record))
    (base/'analysis').mkdir()
    (base/'analysis/summary.json').write_text(json.dumps({'rollout_phase':'final', 'controls':{'gpu':'gpu'},
        'results':{'B16':{'success':{'rate':.9}, 'timing':{'pipeline':{'median_ms':10}, 'peak_allocated_bytes':2**30}}}}))
    (base/'adaptation').mkdir()
    (base/'adaptation/summary.json').write_text(json.dumps({'validation':[{'step':500,'loss':.4}]}))
    result=dashboard.snapshot(tmp_path,False)
    assert result['points']==[]
    assert result['evaluation']['final'][0]['success']==90
    assert result['evaluation']['benchmarks'][0]['memory']==1
    assert result['evaluation']['learning'][0]['steps']==500
    assert result['evaluation']['runs'][0]['source'].endswith('run.json')


def test_active_candidate_preserves_proposal_timeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(dashboard.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='active'))
    journal=tmp_path/'.rsi/events'; journal.mkdir(parents=True)
    events=[{'kind':'attempt','attempt':'n1','outer':0},
            {'kind':'proposal_rejected','attempt':'n1','reason':'duplicate','retry':1},
            {'kind':'proposal_accepted','attempt':'n1','proposal':{'structural_change':'adapter'}}]
    for i,e in enumerate(events): (journal/f'{i:08}.json').write_text(json.dumps(e))
    log=tmp_path/'.rsi/attempts/n1/probe/stderr.log'; log.parent.mkdir(parents=True)
    log.write_text('Training: 50/500 [')
    row=dashboard.snapshot(tmp_path)['attempts'][0]
    assert row['active'] and row['status']=='evaluating screen'
    assert len(row['timeline'])==3 and row['rejections'][0]['reason']=='duplicate'


def test_archived_study_is_read_only_and_not_mixed_into_active(tmp_path):
    active=tmp_path/'.rsi/events';active.mkdir(parents=True)
    archived=tmp_path/'.rsi-archive-pil-test/events';archived.mkdir(parents=True)
    event={'kind':'attempt','attempt':'old1','outer':0}
    (archived/'00000000.json').write_text(json.dumps(event))
    assert dashboard.snapshot(tmp_path,False)['attempts']==[]
    old=dashboard.snapshot(tmp_path,False,'.rsi-archive-pil-test')
    assert old['service']=='archived'
    assert old['attempts'][0]['id']=='old1'
    import pytest
    with pytest.raises(ValueError): dashboard.snapshot(tmp_path,False,'../../tmp')


def test_worker_plan_available_before_acceptance_without_tool_output(tmp_path):
    journal = tmp_path / '.rsi/events'; journal.mkdir(parents=True)
    (journal/'000.json').write_text(json.dumps({'kind':'attempt','attempt':'n0004','outer':0}))
    log = tmp_path/'.rsi/attempts/n0004/proposal_1/discovery/stdout.log'
    log.parent.mkdir(parents=True)
    proposal = {'hypothesis':'Earlier layers preserve useful detail.', 'structural_change':'Fuse depth states.'}
    records = [{'item':{'type':'reasoning','text':'private'}},
               {'item':{'type':'command_execution','text':'tool output'}},
               {'item':{'type':'agent_message','text':'I will test depth fusion.'}},
               {'item':{'type':'agent_message','text':json.dumps(proposal)}}]
    log.write_text('\n'.join(json.dumps(r) for r in records)+'\n{partial')
    before = log.read_bytes()
    row = dashboard.snapshot(tmp_path,False)['attempts'][0]
    assert row['proposal'] is None
    assert row['draft_proposal'] == proposal
    assert row['proposal_history'][0]['messages'] == ['I will test depth fusion.']
    assert log.read_bytes() == before
    assert dashboard.proposal_history(tmp_path/'.rsi', '../../outside') == []


def test_training_curve_uses_exact_tqdm_step_and_refreshes(tmp_path):
    path = tmp_path/'stderr.log'
    path.write_text('Training: 1050/1500 [elapsed]\rINFO step:1K loss:0.391 action_loss:0.250 wm_loss:0.141\nTraining: 1093/1500 [elapsed]')
    curve = dashboard.training_curve(path)
    assert curve['step'] == 1093 and curve['total'] == 1500
    assert curve['curve'] == [{'step':1050,'loss':.391,'action_loss':.250,'wm_loss':.141}]
    with path.open('a') as handle:
        handle.write('\rTraining: 1100/1500 [elapsed]\rINFO step:1K loss:0.381\n')
    assert dashboard.training_curve(path)['curve'][-1] == {'step':1100,'loss':.381}


def test_rollout_progress_ignores_partial_csv_rows(tmp_path):
    path=tmp_path/'episodes.csv'
    path.write_text('task_name,success,status\nfirst task,0,completed\nfirst task,1,completed\nsecond task,\n')
    p=dashboard.rollout_progress(path, {'arguments':{'tasks':[0,1], 'episodes':'10'}})
    assert (p['completed'],p['total'],p['successes']) == (2,20,1)
    assert p['curve'][-1]['success_rate'] == 50
    assert p['last_task'] == 'first task'


def test_cloud_training_overrides_stale_local_confirmation(tmp_path):
    local = tmp_path/'outputs/confirmation/n0008-20261001/status.json'
    local.parent.mkdir(parents=True)
    local.write_text(json.dumps({'arm':'RSI-n0008','status':'running','stage':'training'}))
    before = local.read_bytes()
    cloud = tmp_path/'outputs/cloud/remote-job.json'
    cloud.parent.mkdir(parents=True)
    cloud.write_text(json.dumps({'status':'training'}))
    run = tmp_path/'studies/evaluation/training/cloud'
    run.mkdir(parents=True)
    (run/'run.json').write_text(json.dumps({'variant':'RSI-n0008-5090','requested_steps':10000}))
    (run/'metrics.jsonl').write_text(json.dumps({'phase':'training','step':5150,'loss':.3})+'\n')
    result = dashboard.confirmation_live(tmp_path)
    assert result['arm'] == 'RSI-n0008-5090'
    assert result['progress']['step'] == 5150
    assert result['progress']['total'] == 10000
    assert result['progress']['cloud']
    assert not result['progress']['completed']
    cloud.write_text(json.dumps({'status':'failed'}))
    assert dashboard.confirmation_live(tmp_path)['status'] == 'failed'
    assert local.read_bytes() == before


def test_cloud_retry_does_not_reuse_interrupted_run_progress(tmp_path):
    cloud = tmp_path/'outputs/cloud/remote-job.json'
    cloud.parent.mkdir(parents=True)
    cloud.write_text(json.dumps({'status':'preparing', 'training_label':'RSI-n0008-5090-r2'}))
    old = tmp_path/'studies/evaluation/training/old'
    old.mkdir(parents=True)
    (old/'run.json').write_text(json.dumps({'variant':'RSI-n0008-5090', 'requested_steps':10000}))
    (old/'metrics.jsonl').write_text(json.dumps({'phase':'training','step':5840,'loss':.3})+'\n')
    result = dashboard.confirmation_live(tmp_path)
    assert result['arm'] == 'RSI-n0008-5090-r2'
    assert result.get('progress', {}).get('step', 0) != 5840
