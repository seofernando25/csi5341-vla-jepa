"""Transfer integrity checks; no models or GPU execution."""

import pytest

from evaluation.common import file_hash
from evaluation.portability import checked_path, verify, verify_execution_sources


def test_bundle_detects_same_size_tampering(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("abc")
    manifest = {"files": {"artifact.txt": {"bytes": 3, "sha256": file_hash(path)}}}
    assert verify(tmp_path, manifest) == 1
    path.write_text("xyz")
    with pytest.raises(ValueError, match="differs"):
        verify(tmp_path, manifest)


def test_bundle_rejects_escape_and_symlink(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("abc")
    (tmp_path / "alias").symlink_to(path)
    for name in ["../outside", str(path), "alias"]:
        with pytest.raises(ValueError):
            checked_path(tmp_path, name)


def test_transfer_rejects_unmeasured_source(tmp_path):
    names = [
        "pyproject.toml",
        "uv.lock",
        "evaluation/common.py",
        "evaluation/models.py",
        "evaluation/run.py",
        "evaluation/libero_setup.py",
        "evaluation/protocol.json",
        "evaluation/sources.json",
        "evaluation/core_benchmark.py",
        "src/plugin.py",
    ]
    manifest = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        manifest[name] = file_hash(path)
    record = {"experiment": "core_benchmark", "environment": {"source_manifest": manifest}}
    verify_execution_sources(tmp_path, record)
    (tmp_path / "src/plugin.py").write_text("changed")
    with pytest.raises(ValueError, match="Source differs"):
        verify_execution_sources(tmp_path, record)
