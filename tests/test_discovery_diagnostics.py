import json
from pathlib import Path

from rsi import discovery_diagnostics as diagnostics


def test_exports_completed_prefix_without_private_output_or_other_studies(tmp_path):
    state = tmp_path/'.rsi'
    log = state/'attempts/n0001/probe/stderr.log'; log.parent.mkdir(parents=True)
    log.write_text('private command and path\nTraining: 1050/1500 [elapsed]\rINFO step:1K loss:0.4 step_s:0.8 mem_gb:7.5\n')
    future = state/'attempts/n0002/probe/stderr.log'; future.parent.mkdir(parents=True)
    future.write_text('Training: 50/500 [elapsed]\rINFO loss:0.1\n')
    events = [{'kind':'outcome','seq':4,'attempt':'n0001','node':{'status':'ok','metrics':{'eval_loss':.3,'hardware':{'gpu':'RTX 3090','peak_allocated_bytes':1024,'private':'excluded'}}}}]
    out = tmp_path/'evidence'
    before = log.read_bytes()
    records = diagnostics.prepare_evidence(state,events,out)
    assert len(records)==1 and records[0]['id']=='n0001'
    assert records[0]['inference_latency_ms'] is None
    assert json.loads((out/'n0001-screen-0.jsonl').read_text()) == {'step':1050,'total':1500,'loss':.4,'step_s':.8,'mem_gb':7.5}
    assert 'private' not in (out/'index.json').read_text()
    assert log.read_bytes()==before
    assert not list(out.glob('n0002*'))


def test_samples_preserve_training_components():
    rows = diagnostics.logged_samples('Training: 100/500 [elapsed]\rINFO loss:0.5 action_loss:0.3 wm_loss:0.2 lr:1e-5\n')
    assert rows[0]['action_loss']==.3 and rows[0]['wm_loss']==.2


def test_explicit_amendment_preserves_scientific_freeze(tmp_path, monkeypatch):
    import shutil
    import pytest
    from rsi.runner import Runner
    from types import SimpleNamespace
    monkeypatch.setattr("rsi.runner.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="test-revision"))
    root = Path(__file__).parents[1]
    repo = tmp_path/'repo'; repo.mkdir()
    for name in ('rsi','src'):
        shutil.copytree(root/name, repo/name)
    shutil.copyfile(root/'pyproject.toml',repo/'pyproject.toml')
    runner = Runner(repo,repo/'.rsi',synthetic=True)
    runner.initialize(repo/'rsi/config.json')
    baseline = runner.events('initialized')[0]
    prompt = repo/'rsi/prompts/discovery.txt'
    prompt.write_text(prompt.read_text()+'\nReport completed checks only.\n')
    with pytest.raises(ValueError,match='frozen config/harness'):
        runner.verify()
    runner.update_discovery_context('Operator authorized richer current-study diagnostics')
    assert runner.verify()['config_hash']==baseline['config_hash']
    assert runner.events('discovery_context_updated')[0]['changed_files']==['prompts/discovery.txt']
    evaluator = repo/'rsi/evaluator.py'; evaluator.write_text(evaluator.read_text()+'\n# modified\n')
    with pytest.raises(ValueError,match='approved context'):
        runner.update_discovery_context('Cannot amend evaluator')
