"""An unpacked cloud source bundle retains provenance without Git metadata."""
from evaluation import common


def test_environment_in_source_bundle_without_git(tmp_path, monkeypatch):
    import torch

    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "bundle"\n')
    (tmp_path / 'uv.lock').write_text('version = 1\n')
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    result = common.environment()
    assert result['git_revision'] is None
    assert result['source_manifest'] == {
        name: common.file_hash(tmp_path / name)
        for name in ['pyproject.toml', 'uv.lock']
    }
    assert result['cuda_available'] is False
