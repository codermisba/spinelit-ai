"""Hub-fallback tests for the standalone segmentation checkpoint.

Hosted deployments exclude ``checkpoints/*.pth``, so the loaders must use the
file downloaded from ``MODEL_ID``.  A regression here silently disables masks
in production, so it is covered explicitly.
"""

import pytest
import torch
from pathlib import Path

import segmentation
from segmentation import UNet, load_segmentation_model


@pytest.fixture
def hub_checkpoint(tmp_path):
    model = UNet(num_classes=4, base_channels=4)
    path = tmp_path / "segmentation.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": {
                "num_classes": 4,
                "base_channels": 4,
                "image_size": 128,
                "version": "test-v1",
                "val_dice": 0.5,
            },
        },
        path,
    )
    return path


def test_loader_uses_downloaded_hub_checkpoint(hub_checkpoint, monkeypatch):
    monkeypatch.setattr(segmentation, "_hub_segmentation_path",
                        lambda: hub_checkpoint)

    model, device, path, meta = load_segmentation_model(
        Path("definitely_missing_local_segmentation.pth"))

    assert Path(path) == hub_checkpoint
    assert meta["version"] == "test-v1"
    assert meta["val_dice"] == 0.5
    assert not model.training


def test_loader_raises_when_hub_also_unavailable(monkeypatch):
    monkeypatch.setattr(segmentation, "_hub_segmentation_path", lambda: None)

    with pytest.raises(FileNotFoundError, match="No segmentation checkpoint"):
        load_segmentation_model(Path("definitely_missing_local_segmentation.pth"))
