"""Refuse paid diagnostic execution when export, time or money guards fail."""

import copy

import pytest

from scripts.recovery_export import job_identity
from scripts.recovery_parent_native import readiness


def context():
    job = {'study': 'query', 'recipe_sha256': 'fixed', 'started_at': 100, 'status': 'completed'}
    export = {'status': 'verified_completion_review', 'at': 990,
              'instance_id': 7, 'total_spent_usd': 11}
    review = {'job_identity': job_identity(job), 'until': 4000}
    rental = {'instance_id': 7, 'planned_cleanup_epoch': 5000, 'offer': {'dph_total': .53}}
    return job, export, review, rental


def test_verified_bounded_completion_is_eligible():
    readiness(*context(), 1000)


@pytest.mark.parametrize('item,field,value,reason', [
    (0, 'status', 'training', 'not completed'),
    (1, 'status', 'monitoring', 'verified final-export'),
    (1, 'at', 800, 'fresh'),
    (2, 'job_identity', 'previous-job', 'another job'),
    (2, 'until', 2100, 'completion-review time'),
    (1, 'instance_id', 8, 'identities differ'),
    (1, 'total_spent_usd', 13.2, 'spending reserve'),
    (3, 'planned_cleanup_epoch', 2190, 'completion-review time'),
])
def test_unsafe_execution_contexts_are_refused(item, field, value, reason):
    values = copy.deepcopy(context())
    values[item][field] = value
    with pytest.raises(ValueError, match=reason):
        readiness(*values, 1000)
