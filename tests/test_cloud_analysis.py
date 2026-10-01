import pytest
from evaluation.cloud_analysis import validation


def rows():
    return [{'phase':'validation','step':10000,'batch':i,'samples':2,
             'loss':i/100,'action_loss':i/200,'wm_loss':i/200} for i in range(100)]


def test_validation_uses_all_fixed_samples():
    assert validation(rows())[0]['loss'] == pytest.approx(.495)
    assert validation(rows())[0]['samples'] == 200


def test_partial_or_duplicate_validation_is_rejected():
    with pytest.raises(ValueError): validation(rows()[:-1])
    data=rows();data[-1]['batch']=0
    with pytest.raises(ValueError): validation(data)
