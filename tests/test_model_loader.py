from pathlib import Path

import pytest

import model_loader


def make_snapshot(path: Path):
    path.mkdir(parents=True)
    (path / "model_index.json").write_text("{}", encoding="ascii")
    return str(path)


def test_explicit_valid_model_path(monkeypatch, tmp_path):
    snapshot = make_snapshot(tmp_path / "explicit")
    monkeypatch.setenv("IDM_MODEL_PATH", snapshot)
    monkeypatch.delenv("IDM_ALLOW_REMOTE_DOWNLOAD", raising=False)
    assert model_loader._resolve_model_source() == (snapshot, True)


def test_automatic_runpod_cache_snapshot_discovery(monkeypatch, tmp_path):
    snapshot = make_snapshot(tmp_path / "runpod-cache")
    monkeypatch.delenv("IDM_MODEL_PATH", raising=False)
    monkeypatch.delenv("IDM_ALLOW_REMOTE_DOWNLOAD", raising=False)
    monkeypatch.setattr(model_loader, "RUNPOD_CACHE_SNAPSHOT", snapshot)
    assert model_loader._resolve_model_source() == (snapshot, True)


def test_missing_local_snapshot_with_downloads_disabled(monkeypatch, tmp_path):
    explicit = tmp_path / "missing-explicit"
    runpod = tmp_path / "missing-runpod"
    hf_home = tmp_path / "missing-hf-home"
    monkeypatch.setenv("IDM_MODEL_PATH", str(explicit))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("IDM_ALLOW_REMOTE_DOWNLOAD", raising=False)
    monkeypatch.setattr(model_loader, "RUNPOD_CACHE_SNAPSHOT", str(runpod))
    with pytest.raises(RuntimeError) as error:
        model_loader._resolve_model_source()
    message = str(error.value)
    assert all(str(path) in message for path in (explicit, runpod, hf_home))
    assert "model_index.json" in message
    assert "RunPod Model Cache" in message
    assert "Network Volume" in message


def test_remote_fallback_only_when_explicitly_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("IDM_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing-hf-home"))
    monkeypatch.setenv("IDM_ALLOW_REMOTE_DOWNLOAD", "true")
    monkeypatch.setattr(
        model_loader, "RUNPOD_CACHE_SNAPSHOT", str(tmp_path / "missing-runpod")
    )
    assert model_loader._resolve_model_source() == (model_loader.MODEL_ID, False)


def test_preprocessor_validation_rejects_git_lfs_pointer(tmp_path):
    pointer = tmp_path / "parsing_atr.onnx"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789\nsize 123456789\n",
        encoding="ascii",
    )
    assert "Git LFS pointer" in model_loader._preprocessor_problem(str(pointer))


def test_preprocessor_validation_rejects_small_file(tmp_path):
    small = tmp_path / "parsing_lip.onnx"
    small.write_bytes(b"not an ONNX model")
    assert "too small" in model_loader._preprocessor_problem(str(small))


def test_preprocessor_validation_accepts_nontrivial_binary(tmp_path, monkeypatch):
    binary = tmp_path / "body_pose_model.pth"
    binary.write_bytes(b"binary model data")
    monkeypatch.setattr(model_loader, "PREPROCESSOR_MIN_BYTES", 8)
    assert model_loader._preprocessor_problem(str(binary)) is None
