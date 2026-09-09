"""Horizontal-flip augmentation (image + landmark x together) tests."""

import numpy as np
import torch

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