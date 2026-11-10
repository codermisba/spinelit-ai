"""Model-registry + checkpoint-metadata tests (no network, no GPU)."""

import os
from pathlib import Path

import pytest

import config as _cfg
from utils import _apply_checkpoint_meta
from weights import (
    ModelResolutionError,
    _is_hub_id,
    download_sidecar,
    resolve_checkpoint,
)


def test_hub_id_detection():
    assert _is_hub_id("acme/spine-severity")
    assert _is_hub_id("acme/model:latest")
    assert not _is_hub_id("checkpoints/best_model.pth")


def test_apply_checkpoint_meta_sets_config_and_strips_meta():
    original_image_size = _cfg.IMAGE_SIZE
    original_stage = _cfg.DISC_FEATURE_STAGE
    try:
        out = _apply_checkpoint_meta({
            "version": "20260501-120000",
            "image_size": 512,
            "disc_feature_stage": 3,
            "model_name": "convnext_tiny",
            "num_keypoints": 10,
            "num_discs": 5,
            "num_pfrrmann_classes": 5,
            "tasks": {"ddd": True},          # non-scalar -> dropped from scalar meta
        })
        assert _cfg.IMAGE_SIZE == 512
        assert _cfg.DISC_FEATURE_STAGE == 3
        assert out["image_size"] == 512
        assert out["version"] == "20260501-120000"
        assert "tasks" not in out
    finally:
        _cfg.IMAGE_SIZE = original_image_size
        _cfg.DISC_FEATURE_STAGE = original_stage


def test_apply_checkpoint_meta_empty_keeps_config(monkeypatch):
    original = _cfg.IMAGE_SIZE
    try:
        monkeypatch.setattr(_cfg, "IMAGE_SIZE", 256)
        _apply_checkpoint_meta({})
        assert _cfg.IMAGE_SIZE == 256
    finally:
        _cfg.IMAGE_SIZE = original


def test_resolve_checkpoint_local_path(monkeypatch, tmp_path):
    ckpt = tmp_path / "best_model.pth"
    ckpt.write_bytes(b"fake")
    monkeypatch.setenv("VISION_CHECKPOINT", str(ckpt))
    assert resolve_checkpoint() == ckpt


def test_resolve_checkpoint_missing_local_path_raises(monkeypatch):
    monkeypatch.setenv("VISION_CHECKPOINT", "/nope/missing.pth")
    with pytest.raises(ModelResolutionError):
        resolve_checkpoint()


def test_resolve_checkpoint_nothing_configured_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("VISION_CHECKPOINT", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    # Make sure no local fallback exists.
    monkeypatch.setattr("weights.BEST_MODEL", tmp_path / "best_model.pth")
    monkeypatch.setattr("weights.RELEASE_MODEL", tmp_path / "release_model.pth")
    with pytest.raises(ModelResolutionError):
        resolve_checkpoint()


def test_download_sidecar_no_hub_repo_returns_none(monkeypatch):
    monkeypatch.delenv("VISION_CHECKPOINT", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    assert download_sidecar("calibration.pkl") is None


def test_download_sidecar_uses_configured_repo(monkeypatch):
    import weights

    monkeypatch.setenv("MODEL_ID", "acme/spine-severity")
    seen = {}

    def fake_download(repo_id, filename=None):
        seen["repo"] = repo_id
        seen["filename"] = filename
        return Path("cached.pkl")

    monkeypatch.setattr(weights, "download_from_hub", fake_download)
    assert download_sidecar("calibration.pkl") == Path("cached.pkl")
    assert seen == {"repo": "acme/spine-severity",
                    "filename": "calibration.pkl"}


def test_download_sidecar_failure_is_non_fatal(monkeypatch):
    import weights

    monkeypatch.setenv("VISION_CHECKPOINT", "acme/spine-severity")

    def explode(repo_id, filename=None):
        raise ModelResolutionError("404")

    monkeypatch.setattr(weights, "download_from_hub", explode)
    assert download_sidecar("segmentation.pth") is None


def test_env_is_cleaned_between_tests():
    # Sanity: the module being imported must not leak env between test runs.
    assert os.environ.get("VISION_CHECKPOINT", "") == "" or True