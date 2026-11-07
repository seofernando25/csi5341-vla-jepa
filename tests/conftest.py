"""Shared test fixtures."""

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def ample_disk(monkeypatch):
    """Keep the RSI free-disk guard independent of the host's temp filesystem.

    Tests that exercise the guard patch `disk_usage` again with a low value.
    """
    monkeypatch.setattr(
        "rsi.runner.shutil.disk_usage", lambda path: SimpleNamespace(free=1024**4)
    )
