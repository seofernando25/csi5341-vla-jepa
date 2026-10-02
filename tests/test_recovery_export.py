import subprocess
import sys

import pytest

from scripts.recovery_export import transfer_with_updates


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
