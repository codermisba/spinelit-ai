"""
Thin-line anatomical overlay (landmark-derived schematic) tests.

The model predicts 10 keypoints, NOT a segmentation mask. The overlay in
utils.py draws thin (1 px) vertebra-body outlines, a column axis and
horizontal offset lines derived only from those keypoints.
"""

import numpy as np
from PIL import Image

from utils import (
    draw_landmarks,
    draw_thin_anatomy_overlay,
    BODY_OUTLINE_COLOUR,
    ANATOMY_LINE_WIDTH,
)


def _dummy_coords(size=512):
    pts = []
    for i in range(5):
        pts.append([size / 2, 60 + i * 80])      # L1..L5 centres
    for i in range(5):
        pts.append([size / 2, 100 + i * 80])     # disc centres
    return np.array(pts, dtype=np.float32) / size


def _blank_image(size=512):
    return Image.new("RGB", (size, size), (20, 20, 40))


def test_thin_line_width_is_1():
    assert ANATOMY_LINE_WIDTH == 1


def test_overlay_draws_thin_outlines_on_original_size():
    img = _blank_image()
    out = np.asarray(draw_thin_anatomy_overlay(img, _dummy_coords()))
    blank = np.asarray(img)
    assert out.shape == blank.shape
    changed = (out != blank).any(axis=-1).sum()
    assert changed > 500, "thin-line overlay must render lines"

    # body-outline colour present (thin blue rectangle edges)
    target = np.array(BODY_OUTLINE_COLOUR, dtype=np.uint8)
    assert (out.reshape(-1, 3) == target).all(axis=-1).sum() > 100


def test_overlay_is_landmark_derived_not_segmentation():
    """Verify it uses only keypoints: shifting centres shifts the outlines."""
    img = _blank_image()
    base = np.asarray(draw_thin_anatomy_overlay(img, _dummy_coords()))
    shifted = _dummy_coords()
    shifted[:, 0] = (shifted[:, 0] + 0.1) % 1.0   # shift all centres right
    alt = np.asarray(draw_thin_anatomy_overlay(img, shifted))
    assert (base != alt).any(), "overlay must track the detected keypoints"


def test_draw_landmarks_renders_with_and_without_anatomy():
    img = _blank_image()
    coords = _dummy_coords()
    with_anat = draw_landmarks(img, coords, confidence=[0.9] * 10,
                               anatomy_lines=True)
    without = draw_landmarks(img, coords, confidence=[0.9] * 10,
                             anatomy_lines=False)
    assert with_anat.size == img.size
    assert without.size == img.size
    assert np.asarray(with_anat).tolist() != np.asarray(without).tolist()


def test_degenerate_coords_does_not_crash():
    img = _blank_image()
    coords = np.ones((10, 2), dtype=np.float32) * 0.5   # all points identical
    out = draw_thin_anatomy_overlay(img, coords)
    assert out.size == img.size