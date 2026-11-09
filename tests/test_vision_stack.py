"""
tests/test_vision_stack.py
==========================

Regression tests for the rebuilt vision stack:

- ConvNeXt + FPN heatmap model: shapes, soft-argmax range, confidence
- ImageNet normalisation consistency
- Grad-CAM on the FPN feature map
- Standalone U-Net segmentation + mask-to-box conversion
- Landmark-derived blocks and composed rendering
- EvidenceCard DDD gating (never fabricate grades)
- VisionEngine end-to-end with a tiny checkpoint (landmark detections,
  DDD reported unavailable)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

import config as _cfg
from config import DISC_LEVELS, NUM_DISCS, NUM_KEYPOINTS, NUM_PFRRMANN_CLASSES
from schemas import build_evidence_card
from utils import (
    draw_blocks,
    get_transform,
    landmark_boxes,
    render_analysis,
)


# ----------------------------------------------------------------------
# Model / localizer
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    """Small random-weight model (fused FPN dim 64, 64x64 input)."""
    from model import SpineFoundationModel

    old = (_cfg.MODEL_NAME, _cfg.PRETRAINED, _cfg.FPN_DIM, _cfg.IMAGE_SIZE)
    _cfg.MODEL_NAME = "convnext_femto"
    _cfg.PRETRAINED = False
    _cfg.FPN_DIM = 64
    _cfg.IMAGE_SIZE = 64
    try:
        model = SpineFoundationModel()
        model.eval()
        yield model
    finally:
        (_cfg.MODEL_NAME, _cfg.PRETRAINED, _cfg.FPN_DIM,
         _cfg.IMAGE_SIZE) = old


def test_heatmap_model_shapes_and_ranges(tiny_model):
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = tiny_model(x)

    assert out["coords"].shape == (2, NUM_KEYPOINTS * 2)
    assert out["heatmaps"].shape[:2] == (2, NUM_KEYPOINTS)
    assert out["localization_conf"].shape == (2, NUM_KEYPOINTS)
    assert out["ddd_logits"].shape == (2, NUM_DISCS, NUM_PFRRMANN_CLASSES)
    assert out["fmap"].shape[0] == 2 and out["fmap"].dim() == 4

    coords = out["coords"].view(-1, NUM_KEYPOINTS, 2)
    assert float(coords.min()) >= 0.0 and float(coords.max()) <= 1.0
    conf = out["localization_conf"]
    assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0
    peak = out["heatmap_peak"]
    assert peak.shape == (2, NUM_KEYPOINTS)


def test_disc_centers_argument_is_used(tiny_model):
    x = torch.randn(1, 3, 64, 64)
    centers = torch.rand(1, NUM_DISCS, 2) * 0.5 + 0.25
    with torch.no_grad():
        out = tiny_model(x, disc_centers=centers)
    assert out["ddd_logits"].shape == (1, NUM_DISCS, NUM_PFRRMANN_CLASSES)
    assert torch.isfinite(out["ddd_logits"]).all()


def test_heatmap_soft_argmax_follows_shifted_signal(tiny_model):
    """A brighter/structured input must move the predicted points, proving
    the head is spatial rather than a constant regression."""
    torch.manual_seed(0)
    a = torch.zeros(1, 3, 64, 64)
    b = torch.zeros(1, 3, 64, 64)
    a[:, :, 8:20, 8:24] = 3.0
    b[:, :, 40:56, 36:52] = 3.0
    with torch.no_grad():
        pa = tiny_model(a)["coords"].view(NUM_KEYPOINTS, 2)
        pb = tiny_model(b)["coords"].view(NUM_KEYPOINTS, 2)
    assert not torch.allclose(pa, pb, atol=1e-5)


def test_soft_argmax_temperature_reads_the_peak():
    """Sharpening must make soft-argmax land on the peak, not the centre.

    This is the regression test for the coordinate-collapse bug seen with a
    real trained checkpoint: heatmap peaks were correct but plain softmax on
    small logits returned the image centre.
    """
    from model import _soft_argmax_2d

    logits = torch.zeros(1, 1, 32, 32)
    logits[0, 0, 8, 24] = 3.0            # peak at x=24, y=8
    with torch.no_grad():
        cold, _ = _soft_argmax_2d(logits, temperature=1.0)
        sharp, _ = _soft_argmax_2d(logits, temperature=20.0)

    target = torch.tensor([(24 + 0.5) / 32, (8 + 0.5) / 32])
    cold_err = float((cold[0] - target).abs().max())
    sharp_err = float((sharp[0] - target).abs().max())
    assert sharp_err < 0.05
    assert sharp_err < cold_err

    # The path must stay differentiable for the coordinate loss.
    logits.requires_grad_(True)
    coords, _ = _soft_argmax_2d(logits, temperature=20.0)
    coords.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_get_transform_applies_imagenet_normalisation():
    transform = get_transform()
    gray = Image.new("RGB", (32, 32), (128, 128, 128))
    tensor = transform(gray)
    expected = (128 / 255 - np.array(_cfg.IMAGENET_MEAN)) / np.array(
        _cfg.IMAGENET_STD
    )
    assert tensor.shape == (3, _cfg.IMAGE_SIZE, _cfg.IMAGE_SIZE)
    assert np.allclose(tensor.mean(dim=(1, 2)).numpy(), expected, atol=1e-3)


# ----------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------

def test_gradcam_hooks_fpn_and_returns_input_size_heatmap(tiny_model):
    from explainability import GradCAM

    cam = GradCAM(tiny_model)
    try:
        x = torch.randn(1, 3, 64, 64)
        heat = cam.cam_for(x, disc_index=2, class_index=3)
        cls, top = cam.cam_for_top_class(x, disc_index=1)
    finally:
        cam.close()

    assert heat.shape == (64, 64)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0 + 1e-6
    assert top.shape == (64, 64)
    assert cls in range(NUM_PFRRMANN_CLASSES)


def test_gradcam_localization_mode_returns_input_size_heatmap(tiny_model):
    """Without Pfirrmann labels Grad-CAM must still be able to explain a
    *landmark* (clearly a localization map, not a grade explanation)."""
    from explainability import GradCAM

    cam = GradCAM(tiny_model)
    try:
        x = torch.randn(1, 3, 64, 64)
        heat = cam.cam_for_localization(x, keypoint_index=3)
    finally:
        cam.close()

    assert heat.shape == (64, 64)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0 + 1e-6
    assert float(heat.max()) > 0.0


# ----------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------

def test_unet_forward_shape():
    from segmentation import UNet

    model = UNet(num_classes=4, base_channels=8)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 4, 64, 64)


def test_mask_boxes_finds_components_and_class_names():
    from segmentation import mask_boxes

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:20, 10:30] = 1          # vertebra blob
    mask[40:50, 45:70] = 2          # disc blob
    boxes = mask_boxes(mask, min_area=10)

    assert boxes, "expected at least two components"
    classes = {b["class_id"] for b in boxes}
    assert classes == {1, 2}
    for box in boxes:
        assert box["source"] == "segmentation"
        x0, y0, x1, y1 = box["box"]
        assert 0 <= x0 <= x1 < 80 and 0 <= y0 <= y1 < 80
        names = list(_cfg.SEG_CLASS_NAMES)
        assert box["class_name"] == names[box["class_id"]]


def test_segmentation_engine_reports_unavailable_without_checkpoint(tmp_path):
    from segmentation import SegmentationEngine

    engine = SegmentationEngine(tmp_path / "missing.pth")
    assert engine.available is False
    status = engine.status()
    assert status["loaded"] is False
    assert status["error"]
    with pytest.raises(RuntimeError):
        engine.analyze(Image.new("RGB", (32, 32)))


# ----------------------------------------------------------------------
# Blocks / rendering
# ----------------------------------------------------------------------

def _dummy_coords():
    verts = np.column_stack([
        np.linspace(0.42, 0.56, 5),
        np.linspace(0.15, 0.80, 5),
    ])
    discs = np.column_stack([
        np.linspace(0.42, 0.56, 5),
        np.linspace(0.20, 0.75, 5),
    ])
    return np.concatenate([verts, discs], axis=0)


def test_landmark_boxes_cover_all_levels():
    coords = _dummy_coords()
    boxes = landmark_boxes(coords, 200, 300)
    assert len(boxes) == len(_cfg.VERTEBRAE) + len(DISC_LEVELS)
    assert {b["class_name"] for b in boxes} == {"vertebra", "disc"}
    assert {b["source"] for b in boxes} == {"landmark"}
    for item in boxes:
        x0, y0, x1, y1 = item["box"]
        assert x1 > x0 and y1 > y0


def test_render_analysis_composes_without_error():
    image = Image.new("RGB", (120, 180), (25, 25, 25))
    coords = _dummy_coords()
    mask = np.zeros((180, 120), dtype=np.uint8)
    mask[60:90, 40:80] = 1
    out = render_analysis(
        image, coords=coords,
        confidence=np.linspace(0.9, 0.5, NUM_KEYPOINTS),
        segmentation=mask,
        gradcam=np.zeros((180, 120), dtype=np.float32),
        show_blocks=True, show_landmarks=True,
    )
    assert out.size == image.size
    assert np.asarray(out).shape == (180, 120, 3)


def test_draw_blocks_accepts_detector_boxes():
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    boxes = [{
        "label": "L4", "class_name": "vertebra",
        "box": [10, 10, 50, 40], "source": "yolo", "confidence": 0.87,
    }]
    out = draw_blocks(image, boxes=boxes)
    assert out.size == image.size


# ----------------------------------------------------------------------
# EvidenceCard DDD gating
# ----------------------------------------------------------------------

def _card_args():
    coords = _dummy_coords()
    from utils import norm_to_pixels
    pixels = norm_to_pixels(coords, 200, 300)
    geo = [{"level": level, "relative_space": 1.0}
           for level in DISC_LEVELS]
    return dict(
        points=coords,
        localization_conf=np.full(NUM_KEYPOINTS, 0.8),
        pixels=pixels,
        geo=geo,
        ddd_grade=np.full(NUM_DISCS, 3.0),
        ddd_prob=np.full((NUM_DISCS, NUM_PFRRMANN_CLASSES), 0.2),
        ddd_conf=np.full(NUM_DISCS, 0.2),
        image_name="case.jpg",
        image_size=(200, 300),
    )


def test_evidence_card_reports_ddd_unavailable_instead_of_random_grades():
    card = build_evidence_card(**_card_args(), ddd_available=False)
    assert card.ddd_available is False
    assert card.ddd == []
    assert any("DDD (Pfirrmann) grading unavailable" in n for n in card.notes)
    # Landmarks and geometry remain valid.
    assert len(card.landmark_points) == NUM_KEYPOINTS
    assert len(card.geometric_indicators) == len(DISC_LEVELS)
    assert card.image_size == [200, 300]


def test_evidence_card_reports_ddd_when_available():
    card = build_evidence_card(**_card_args(), ddd_available=True)
    assert card.ddd_available is True
    assert len(card.ddd) == len(DISC_LEVELS)
    assert all(f.pfirrmann_grade == 3 for f in card.ddd)


# ----------------------------------------------------------------------
# VisionEngine end-to-end with a tiny checkpoint
# ----------------------------------------------------------------------

def _write_tiny_checkpoint(path, tasks):
    from model import SpineFoundationModel

    model = SpineFoundationModel()
    payload = {
        "epoch": 1,
        "tasks": tasks,
        "meta": {
            "version": "test-tiny",
            "created_at": "2026-01-01T00:00:00",
            "image_size": 64,
            "model_name": "convnext_femto",
            "fpn_dim": 64,
            "num_keypoints": NUM_KEYPOINTS,
            "num_discs": NUM_DISCS,
            "num_pfrrmann_classes": NUM_PFRRMANN_CLASSES,
            "num_seg_classes": _cfg.NUM_SEG_CLASSES,
            "use_imagenet_norm": True,
            "disc_feature_stage": _cfg.DISC_FEATURE_STAGE,
            "tasks": tasks,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


@pytest.fixture
def tiny_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(_cfg, "MODEL_NAME", "convnext_femto")
    monkeypatch.setattr(_cfg, "PRETRAINED", False)
    monkeypatch.setattr(_cfg, "FPN_DIM", 64)
    monkeypatch.setattr(_cfg, "IMAGE_SIZE", 64)
    return _write_tiny_checkpoint(
        tmp_path / "tiny.pth",
        tasks={"ddd": False, "seg": False},
    )


def test_vision_engine_end_to_end_without_ddd(tiny_checkpoint):
    from vision_engine import VisionEngine

    engine = VisionEngine(
        checkpoint=tiny_checkpoint,
        enable_detector=False,
        enable_segmentation=False,
        enable_gradcam=False,
    )
    assert engine.available, engine.error
    assert engine.ddd_available is False

    image = Image.new("RGB", (90, 140), (30, 30, 30))
    result = engine.analyze_full(image, image_name="synthetic.png")

    card = result["card"]
    assert card.ddd_available is False
    assert card.ddd == []
    assert len(card.landmark_points) == NUM_KEYPOINTS
    assert card.image_size == [90, 140]

    # Detector disabled -> landmark-derived blocks, honestly labelled.
    assert result["detection_source"] == "landmark"
    n_landmarks = len(_cfg.VERTEBRAE) + len(DISC_LEVELS)
    assert len(result["detections"]) == n_landmarks
    assert all(d["source"] == "landmark" for d in result["detections"])

    assert result["segmentation"]["available"] is False
    assert result["gradcam"] is None
    assert result["annotated_image"].size == image.size


def test_vision_engine_reports_missing_checkpoint(tmp_path):
    from vision_engine import VisionEngine

    engine = VisionEngine(
        checkpoint=tmp_path / "nope.pth",
        enable_detector=False,
        enable_segmentation=False,
        enable_gradcam=False,
    )
    assert engine.available is False
    assert engine.error

    image = Image.new("RGB", (32, 32), (0, 0, 0))
    result = engine.analyze_full(image)
    assert result["detection_source"] == "none"
    assert result["detections"] == []
    assert result["card"].image_processed is False


# ----------------------------------------------------------------------
# YOLO pseudo-box preparation (pure helpers; no data copy)
# ----------------------------------------------------------------------

def test_yolo_data_yaml_written(tmp_path):
    from yolo_prep import _write_data_yaml

    path = _write_data_yaml(tmp_path)
    text = path.read_text()
    assert "train: train.txt" in text
    assert "val: val.txt" in text
    assert "0: vertebra" in text
    assert "1: disc" in text
