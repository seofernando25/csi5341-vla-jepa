"""Supervisor gates must hold without launching research or using CUDA."""
import datetime as dt
import importlib.util
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location('overnight', Path(__file__).parents[1] / 'scripts/rsi_overnight.py')
overnight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overnight)


def test_expired_deadline_never_launches(monkeypatch):
    commands = []
    monkeypatch.setattr(sys, 'argv', ['overnight', '--until', '2020-01-01T07:00:00-05:00'])
    monkeypatch.setattr(overnight, 'cli', lambda *args, **kwargs: commands.append(args))
    monkeypatch.setattr(overnight, 'gpu_ready', lambda: (_ for _ in ()).throw(AssertionError('expired window')))
    assert overnight.main() == 0
    assert commands == [('stop',), ('export',)]


def test_gpu_query_timeout_means_unavailable(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 10)
    monkeypatch.setattr(overnight.subprocess, 'run', timeout)
    assert not overnight.gpu_ready()


def test_progress_tracks_logs_and_events(tmp_path):
    (tmp_path / 'events').mkdir()
    event = tmp_path / 'events' / '00001.json'
    event.write_text('{}')
    assert overnight.latest_progress(tmp_path) == event.stat().st_mtime
