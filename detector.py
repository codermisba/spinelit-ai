"""
detector.py
===========

Optional YOLO vertebra/disc detector used by the vision engine and API.

Two operating modes
-------------------
1. **Trained YOLO** (``checkpoints/yolo_best.pt``) — true detector output.
   Train it with ``yolo_train.py`` after building pseudo-boxes with
   ``python yolo_prep.py --from_landmarks``.
2. **Landmark fallback** — if no detector checkpoint exists, the API still
   draws vertebra/disc blocks derived from the localizer's keypoints.  This
   is always labelled ``source="landmark"`` so a client can tell the
   difference; it is never presented as detector output.

Classes
-------
    0 vertebra
    1 disc
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config as _cfg
from config import YOLO_CHECKPOINT, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD


def landmark_detections(coords, width: int, height: int) -> list[dict]:
    """Build fallback detections from the 10 keypoints (pixel boxes)."""
    from utils import landmark_boxes

    detections = []
    for item in landmark_boxes(coords, width, height, include_discs=True):
        detections.append({
            "label": item["label"],
            "class_name": item["class_name"],
            "box": item["box"],
            "confidence": None,
            "source": "landmark",
        })
    return detections


def segmentation_detections(mask: np.ndarray) -> list[dict]:
    """Convert a segmentation mask into boxes (true mask-derived detections)."""
    from segmentation import mask_boxes

    detections = []
    for item in mask_boxes(mask):
        detections.append({
            "label": item["class_name"],
            "class_name": item["class_name"],
            "box": item["box"],
            "confidence": None,
            "source": "segmentation",
        })
    return detections


class DetectorEngine:
    """Lazy Ultralytics wrapper (optional, CPU-friendly)."""

    def __init__(self, checkpoint: Optional[str | Path] = None):
        self._model = None
        self._path = None
        self._names: dict[int, str] = {}
        self.error = ""
        self.load(checkpoint)

    def load(self, checkpoint: Optional[str | Path] = None):
        path = Path(checkpoint or YOLO_CHECKPOINT)
        if not path.exists():
            self._model = None
            self._path = None
            self.error = f"No detector checkpoint at {path}"
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(path))
            names = getattr(self._model, "names", None) or {}
            self._names = {int(k): str(v) for k, v in names.items()}
            self._path = path
            self.error = ""
        except Exception as exc:  # noqa: BLE001 - optional component
            self._model = None
            self._path = path
            self.error = f"Failed to load YOLO checkpoint: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "loaded": self.available,
            "checkpoint": str(self._path) if self._path else "none",
            "classes": self._names,
            "error": self.error,
            "fallback": "landmark" if not self.available else None,
        }

    def detect(self, image: Image.Image,
               conf: float = YOLO_CONF_THRESHOLD,
               iou: float = YOLO_IOU_THRESHOLD) -> list[dict]:
        """Run the detector and return pixel-space boxes."""
        if not self.available:
            return []
        rgb = np.asarray(image.convert("RGB"))
        # Ultralytics expects BGR numpy arrays (OpenCV convention).
        bgr = rgb[:, :, ::-1].copy()
        try:
            results = self._model.predict(
                bgr, conf=float(conf), iou=float(iou), verbose=False
            )
        except Exception as exc:  # noqa: BLE001 - never break the API
            self.error = f"Detector inference failed: {exc}"
            return []

        detections: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(xyxy, confs, classes):
                name = self._names.get(int(cls), f"class_{int(cls)}")
                name = "disc" if "disc" in name.lower() else (
                    "vertebra" if "vert" in name.lower() else name
                )
                detections.append({
                    "label": name,
                    "class_name": name,
                    "box": [float(v) for v in box],
                    "confidence": float(score),
                    "source": "yolo",
                })
        return detections


def build_detections(
    coords,
    width: int,
    height: int,
    detector: Optional[DetectorEngine] = None,
    image: Optional[Image.Image] = None,
    segmentation_mask: Optional[np.ndarray] = None,
) -> tuple[list[dict], str]:
    """
    Choose the best available detection source.

    Priority: trained YOLO -> segmentation mask boxes -> landmark-derived.
    Returns ``(detections, source_name)``.
    """
    if detector is not None and detector.available and image is not None:
        dets = detector.detect(image)
        if dets:
            return dets, "yolo"

    if segmentation_mask is not None:
        dets = segmentation_detections(segmentation_mask)
        if dets:
            return dets, "segmentation"

    return landmark_detections(coords, width, height), "landmark"
