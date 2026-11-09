"""Horizontal-flip augmentation (image + landmark x together) tests."""

import numpy as np
import torch
from PIL import Image

from dataset import _maybe_flip_augment


def _deterministic_rng(seed=7):
    return np.random.RandomState(seed)


def test_flip_mirrors_image_and_landmark_x_together():
    img = torch.rand(3, 64, 64)
    coords = torch.tensor([0.8, 0.3, 0.2, 0.6, 0.5, 0.5],
                          dtype=torch.float32)   # x,y,x,y,x,y
    flipped_img, flipped_coords = _maybe_flip_augment(
        img, coords, p=1.0, rng=_deterministic_rng()
    )
    assert torch.allclose(flipped_img, torch.flip(img, dims=[2]))
    assert torch.allclose(flipped_coords[0::2], 1.0 - coords[0::2])
    assert torch.allclose(flipped_coords[1::2], coords[1::2])


def test_flip_keeps_y_coordinates_unchanged():
    coords = torch.tensor([0.1, 0.9, 0.4, 0.7], dtype=torch.float32)
    img = torch.rand(3, 32, 32)
    _, out = _maybe_flip_augment(img, coords, p=1.0,
                                 rng=_deterministic_rng())
    assert torch.allclose(out[1::2], coords[1::2])


def test_no_flip_when_probability_zero():
    img = torch.rand(3, 32, 32)
    coords = torch.tensor([0.3, 0.3, 0.6, 0.6], dtype=torch.float32)
    out_img, out_coords = _maybe_flip_augment(
        img, coords, p=0.0, rng=_deterministic_rng()
    )
    assert torch.equal(out_img, img)
    assert torch.equal(out_coords, coords)


def test_default_transform_tracks_runtime_config_image_size(monkeypatch):
    """The dataset must preprocess at the *runtime* IMAGE_SIZE.

    Regression test: importing the constant directly meant
    ``train.py --image-size N`` (and checkpoint meta applied by
    ``utils.load_model``) did not reach the data pipeline, so training and
    inference silently used different scales.
    """
    import config
    from dataset import SpineDataset

    monkeypatch.setattr(config, "IMAGE_SIZE", 96)
    dataset = SpineDataset()
    out = dataset.transform(Image.new("RGB", (200, 300), color=(128, 128, 128)))
    assert out.shape[-2:] == (96, 96)