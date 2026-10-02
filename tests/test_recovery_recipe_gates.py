import json
import sys

import pytest

from evaluation.common import ROOT, file_hash, read_json
from evaluation import recovery_train, run


@pytest.mark.parametrize('steps,preflight,batch', [(2, False, 8), (21, True, 8), (2, True, 4)])
def test_invalid_engineering_launch_stops_before_model_creation(tmp_path, monkeypatch, steps, preflight, batch):
    recipe = read_json(ROOT / 'evaluation/query_preflight_config.json')
    recipe['batch_size'] = batch
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    argv = ['recovery_train', '--recipe', str(path), '--steps', str(steps),
            '--dataset-root', str(tmp_path), '--architecture-source', str(tmp_path),
            '--checkpoint', str(tmp_path)]
    if preflight:
        argv.append('--preflight')
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as error:
        recovery_train.main()
    assert error.value.code == 2


def test_frozen_r1_recipe_is_unchanged():
    assert file_hash(ROOT / 'evaluation/recovery_config.json') == '0e8cf976a3f532dee0f8aa3bf209d664fbda6d5226d429f7e527941776f84074'


def test_nonlegacy_inference_requires_distinct_smol_label(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run', 'rollout', '--variant', 'B16',
                                    '--buffer-precision', 'native_rope'])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
