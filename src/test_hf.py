import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("huggingface_hub")
pytest.importorskip("maia3")

import load_pretrained


def test_download_maia3_checkpoint_uses_expected_defaults(monkeypatch):
    calls = {}

    def fake_hf_hub_download(repo_id, filename):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return "/tmp/fake-maia3.pt"

    monkeypatch.setattr(load_pretrained, "hf_hub_download", fake_hf_hub_download)

    path = load_pretrained.download_maia3_checkpoint()

    assert path == "/tmp/fake-maia3.pt"
    assert calls == {
        "repo_id": "UofTCSSLab/Maia3-5M",
        "filename": "maia3-5m.pt",
    }
