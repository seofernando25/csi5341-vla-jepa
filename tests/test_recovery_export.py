import subprocess
import sys

import pytest

from scripts.recovery_export import completion_window, job_identity, transfer_with_updates, native_export_directory


def test_transfer_updates_survive_metadata_failure():
    calls = []

    def refresh():
        calls.append(True)
        raise RuntimeError('Temporary metadata failure')

    transfer_with_updates([sys.executable, '-c', 'import time; time.sleep(.08)'],
                          update=refresh, interval=.01, timeout=2)
    assert calls


def test_failed_transfer_cannot_be_reported_as_success():
    with pytest.raises(subprocess.CalledProcessError):
        transfer_with_updates([sys.executable, '-c', 'raise SystemExit(3)'], timeout=2)


def test_stalled_transfer_is_terminated_at_deadline():
    with pytest.raises(subprocess.TimeoutExpired):
        transfer_with_updates([sys.executable, '-c', 'import time; time.sleep(10)'],
                              interval=.01, timeout=.03)


def test_completion_review_cannot_extend_after_restart():
    first = completion_window(1000, 60, 'job-a', None, 4., .5, 20000)
    resumed = completion_window(2500, 60, 'job-a', first, 4.2, .5, 20000)
    assert first['until'] == resumed['until'] == 4600
    expired = completion_window(5000, 60, 'job-a', resumed, 4.5, .5, 20000)
    assert expired['until'] <= 5000


def test_review_respects_money_and_provider_deadline():
    assert completion_window(1000, 60, 'x', None, 13.5, .5, 20000)['until'] == 1000
    assert completion_window(1000, 60, 'x', None, 13.49, .5, 20000)['until'] == pytest.approx(1072)
    assert completion_window(1000, 60, 'x', None, 4., .5, 1100)['until'] == 1040
    assert completion_window(1000, 60, 'x', None, 4., 0., 20000)['until'] == 1000


def test_followup_identity_includes_scientific_registration_and_start():
    first = {'study': 'a', 'recipe_sha256': 'abc', 'started_at': 100}
    assert job_identity(first) != job_identity({**first, 'started_at': 101})
    assert job_identity(first) != job_identity({**first, 'recipe_sha256': 'def'})


@pytest.mark.parametrize('folder', ['run-id', 'cloud-backups', 'cloud-backups/query-r1'])
def test_native_export_accepts_current_and_scoped_followup_backups(folder):
    path = f'outputs/recovery/training/{folder}/train/checkpoints/002000/pretrained_model/model.safetensors'
    assert str(native_export_directory(path)) == f'outputs/recovery/training/{folder}/train/checkpoints/002000'


@pytest.mark.parametrize('path', ['outputs/recovery/training/../secret',
                                'outputs/recovery/training/cloud-backups/unregistered/train/checkpoints/002000/pretrained_model/model.safetensors',
                                'outputs/recovery/training/run/train/checkpoints/not-a-step/training_state/rng_state.safetensors',
                                'outputs/recovery/training/run/train/checkpoints/002000/unrelated/key.txt'])
def test_export_rejects_paths_outside_native_or_registered_backup_layout(path):
    with pytest.raises(ValueError):
        native_export_directory(path)
