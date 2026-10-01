import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('confirmation',Path(__file__).parents[1]/'scripts/confirm_libero.py')
confirmation=importlib.util.module_from_spec(spec);spec.loader.exec_module(confirmation)


def test_checkpoint_selection_uses_validation_and_numeric_checkpoint_names(tmp_path,monkeypatch):
    monkeypatch.setattr(confirmation,'ROOT',tmp_path)
    monkeypatch.setattr(confirmation,'EVIDENCE',tmp_path/'evidence')
    recipe={'seed':42}
    monkeypatch.setattr(confirmation,'plan',lambda:{'recipe':recipe,'selection':'held-out loss'})
    runs=[]
    for index,(step,loss) in enumerate([(1500,.4),(5000,.3),(10000,.3)]):
        run_id=f'run{index}'
        record=tmp_path/'studies/evaluation/training'/run_id/'run.json';record.parent.mkdir(parents=True)
        record.write_text(json.dumps({'run_id':run_id,'recipe':recipe,'variant':'candidate'}))
        (record.parent/'metrics.jsonl').write_text(json.dumps({'phase':'validation','step':step,'batch':0,'samples':200,'loss':loss,'action_loss':loss/2,'wm_loss':loss/2})+'\n')
        weights=tmp_path/'outputs/evaluation/training'/run_id/'train/checkpoints'/f'{step:06d}'/'pretrained_model/model.safetensors'
        weights.parent.mkdir(parents=True);weights.write_bytes(b'checkpoint')
        runs.append(record)
    monkeypatch.setattr(confirmation,'training_records',lambda label:runs)
    chosen=confirmation.select('candidate')
    assert chosen.parent.name=='005000'
    result=json.loads((tmp_path/'evidence/candidate-selection.json').read_text())
    assert result['selected_step']==5000 and result['simulator_scores_used'] is False


def test_retention_preserves_milestones_and_latest_resume_state(tmp_path,monkeypatch):
    monkeypatch.setattr(confirmation,'ROOT',tmp_path)
    monkeypatch.setattr(confirmation,'EVIDENCE',tmp_path/'evidence')
    rows=[]
    for step in [500,1500,2000]:
        path=tmp_path/f'{step:06d}'/'pretrained_model';path.mkdir(parents=True)
        (path/'model.safetensors').write_bytes(b'weights')
        (path.parent/'training_state').mkdir()
        rows.append((step,path))
    monkeypatch.setattr(confirmation,'checkpoint_candidates',lambda label:rows)
    confirmation.cleanup('candidate')
    assert not (tmp_path/'000500').exists()
    assert (tmp_path/'001500/pretrained_model/model.safetensors').exists()
    assert not (tmp_path/'001500/training_state').exists()
    assert (tmp_path/'002000/training_state').exists()
    assert list((tmp_path/'evidence/retention').glob('*.json'))
