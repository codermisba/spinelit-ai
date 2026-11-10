"""
vision_engine.py
================

The *Vision Agent*: wraps the rebuilt Spine Foundation imaging model and
turns a single spine image into a machine-readable :class:`EvidenceCard`
plus an annotated image.

What it produces
----------------
- 10 landmarks (5 vertebral centres + 5 disc centres) from dense heatmaps
  + soft-argmax, with per-point confidence.
- Geometric indicators derived from the landmarks.
- Per-level Pfirrmann DDD grades **only when the checkpoint's DDD head was
  trained** (``dataset/ddd_labels.csv`` existed at training time).  A
  checkpoint without a trained DDD head reports ``ddd_available=False`` and
  never returns random grades.
- Optional detections, in priority order: trained YOLO -> segmentation
  mask boxes -> landmark-derived blocks (each labelled with its source).
- Optional segmentation mask from a standalone U-Net checkpoint or the
  joint segmentation head (only when that head was trained).
- Optional Grad-CAM explanation: a DDD/grade CAM when a trained DDD head
  exists, otherwise an explicitly labelled *localization* CAM.

If no trained checkpoint exists, the engine reports this clearly; the API
returns 503 rather than fabricating results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import weights
from calibration import calibrate_ddd
from detector import DetectorEngine, build_detections
from schemas import EvidenceCard, build_evidence_card
from segmentation import SegmentationEngine
from utils import (
    compute_geometric_indicators,
    decode_outputs,
    load_model,
    norm_to_pixels,
    predict,
    preprocess_image,
    render_analysis,
)
import config as _cfg


class VisionEngine:
    """Runs the imaging model and assembles an EvidenceCard."""

    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        detector_checkpoint: Optional[str | Path] = None,
        segmentation_checkpoint: Optional[str | Path] = None,
        enable_segmentation: bool = True,
        enable_detector: bool = True,
        enable_gradcam: bool = True,
    ):
        self._loaded = False
        self._model = None
        self._device = None
        self._checkpoint_name = None
        self._checkpoint_meta: dict = {}
        self.error = ""
        self.enable_gradcam = bool(enable_gradcam)

        self._detector: Optional[DetectorEngine] = None
        self._segmentation: Optional[SegmentationEngine] = None
        self._detector_requested = enable_detector
        self._segmentation_requested = enable_segmentation
        self._detector_checkpoint = detector_checkpoint
        self._segmentation_checkpoint = segmentation_checkpoint

        self.load(checkpoint)

    def load(self, checkpoint: Optional[str | Path] = None):
        try:
            checkpoint_path = checkpoint or weights.resolve_checkpoint()
            model, device, name, meta = load_model(str(checkpoint_path))
            self._model = model
            self._device = device
            self._checkpoint_name = name
            self._checkpoint_meta = meta or {}
            self._loaded = True
            self.error = ""
        except (FileNotFoundError, weights.ModelResolutionError) as exc:
            self._loaded = False
            self.error = str(exc)
            return
        except Exception as exc:  # noqa: BLE001 - service must still start
            self._loaded = False
            self.error = f"Failed to load vision checkpoint: {exc}"
            return

        if self._detector_requested:
            self._detector = DetectorEngine(self._detector_checkpoint)
        if self._segmentation_requested:
            self._segmentation = SegmentationEngine(self._segmentation_checkpoint)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def checkpoint_name(self) -> str:
        return str(self._checkpoint_name) if self._checkpoint_name else "none"

    @property
    def checkpoint_meta(self) -> dict:
        return dict(self._checkpoint_meta)

    @property
    def model_version(self) -> str:
        v = self._checkpoint_meta.get("version", "")
        if v:
            return str(v)
        return str(self._checkpoint_name) if self._checkpoint_name else ""

    @property
    def tasks(self) -> dict:
        return dict(self._checkpoint_meta.get("tasks") or {
            "localization": True, "ddd": False, "seg": False,
        })

    @property
    def ddd_available(self) -> bool:
        return bool(self.available and self.tasks.get("ddd", False))

    @property
    def detector(self) -> Optional[DetectorEngine]:
        return self._detector

    @property
    def segmentation(self) -> Optional[SegmentationEngine]:
        return self._segmentation

    def status(self) -> dict:
        detector_status = (
            self._detector.status() if self._detector is not None
            else {"loaded": False, "error": "disabled"}
        )
        seg_status = (
            self._segmentation.status() if self._segmentation is not None
            else {"loaded": False, "error": "disabled"}
        )
        return {
            "loaded": self.available,
            "checkpoint": self.checkpoint_name,
            "model_version": self.model_version,
            "image_size": _image_size_from_meta(self._checkpoint_meta),
            "tasks": self.tasks,
            "ddd_available": self.ddd_available,
            "detector": detector_status,
            "segmentation": seg_status,
            "error": self.error,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_model(self, image: Image.Image) -> dict:
        outputs = predict(
            self._model, preprocess_image(image), self._device,
            return_seg=bool(self.tasks.get("seg", False)),
        )
        return decode_outputs(outputs)

    def analyze(self, image: Image.Image, image_name: str = "") -> EvidenceCard:
        card, _ = self.analyze_and_draw(image, image_name=image_name)
        return card

    def analyze_and_draw(
        self,
        image: Image.Image,
        image_name: str = "",
        explain: Optional[bool] = None,
    ) -> tuple[EvidenceCard, Image.Image]:
        """Backwards-compatible helper returning (card, annotated image)."""
        result = self.analyze_full(
            image, image_name=image_name, explain=explain
        )
        return result["card"], result["annotated_image"]

    def analyze_full(
        self,
        image: Image.Image,
        image_name: str = "",
        explain: Optional[bool] = None,
    ) -> dict:
        """
        Run the full vision stack and return every artefact the API needs:

        ``card``, ``annotated_image``, ``detections``, ``detection_source``,
        ``segmentation`` (mask/boxes/available), ``gradcam`` (heatmap/info),
        ``decoded`` (raw arrays).
        """
        if not self.available:
            card = EvidenceCard(
                image_processed=False, image_name=image_name,
                notes=[self.error or "Vision model not loaded."],
            )
            return {
                "card": card,
                "annotated_image": image.convert("RGB"),
                "detections": [],
                "detection_source": "none",
                "segmentation": {"available": False, "source": "none"},
                "gradcam": None,
                "decoded": {},
            }

        decoded = self._run_model(image)
        width, height = image.size
        points = decoded["points"]
        conf = decoded["localization_conf"]
        pixels = norm_to_pixels(points, width, height)
        geo = compute_geometric_indicators(pixels)

        # ---- segmentation (standalone model preferred) -----------------
        segmentation_info = {"available": False, "source": "none"}
        seg_mask = None
        if self._segmentation is not None and self._segmentation.available:
            try:
                seg = self._segmentation.analyze(image)
                seg_mask = seg["mask"]
                segmentation_info = {
                    "available": True,
                    "source": "unet",
                    "class_names": list(_cfg.SEG_CLASS_NAMES),
                    "boxes": seg.get("boxes", []),
                    "version": self._segmentation._meta.get("version", ""),
                }
            except Exception as exc:  # noqa: BLE001 - optional component
                segmentation_info = {
                    "available": False, "source": "unet",
                    "error": str(exc),
                }
        elif self.tasks.get("seg", False) and "seg_mask" in decoded:
            seg_mask = decoded["seg_mask"]
            segmentation_info = {
                "available": True,
                "source": "joint_head",
                "class_names": list(_cfg.SEG_CLASS_NAMES),
                "boxes": [],
            }

        # ---- detections ------------------------------------------------
        detections, detection_source = build_detections(
            points, width, height,
            detector=self._detector,
            image=image,
            segmentation_mask=seg_mask,
            landmark_confidence=conf,
        )

        # ---- DDD card (honest gating) ----------------------------------
        ddd_available = self.ddd_available
        card = build_evidence_card(
            points=points,
            localization_conf=conf,
            pixels=pixels,
            geo=geo,
            ddd_grade=decoded.get("ddd_grade"),
            ddd_prob=decoded.get("ddd_prob"),
            ddd_conf=decoded.get("ddd_conf"),
            image_name=image_name,
            calibrate=calibrate_ddd,
            ddd_available=ddd_available,
            image_size=(width, height),
        )
        card.detections = detections
        card.detection_source = detection_source
        card.segmentation = segmentation_info
        card.model_tasks = self.tasks
        if not detected_all_landmarks(conf):
            card.notes.append(
                "Some landmark confidences are low; inspect the annotated "
                "image before trusting level assignments."
            )

        # ---- Grad-CAM ---------------------------------------------------
        # A trained DDD head gets a *grade* explanation.  Without Pfirrmann
        # labels we still produce a *localization* explanation for the least
        # confident landmark, clearly tagged ``kind="localization"`` so it is
        # never mistaken for a grade/DDD explanation.
        gradcam = None
        want_explain = self.enable_gradcam if explain is None else bool(explain)
        if want_explain:
            gradcam = self._gradcam(image, decoded)

        # ---- annotated image ------------------------------------------
        annotated = render_analysis(
            image,
            coords=points,
            confidence=conf,
            detections=detections if detection_source != "landmark" else None,
            segmentation=seg_mask,
            gradcam=(gradcam or {}).get("heatmap"),
            show_blocks=True,
            show_landmarks=True,
        )

        return {
            "card": card,
            "annotated_image": annotated,
            "detections": detections,
            "detection_source": detection_source,
            "segmentation": segmentation_info,
            "gradcam": gradcam,
            "decoded": decoded,
        }

    def _gradcam(self, image: Image.Image, decoded: dict) -> Optional[dict]:
        try:
            from explainability import GradCAM

            x = preprocess_image(image).unsqueeze(0).to(self._device)
            cam = GradCAM(self._model)
            try:
                # 1) Real Pfirrmann-grade explanation (trained DDD head only).
                if self.ddd_available:
                    grades = np.asarray(decoded.get("ddd_grade", []),
                                        dtype=float)
                    if grades.size == 0:
                        return None
                    disc_index = int(np.argmax(grades))
                    class_index = int(np.clip(round(grades[disc_index]) - 1,
                                              0, 4))
                    heatmap = cam.cam_for(x, disc_index, class_index)
                    from config import DISC_LEVELS, PFRRMANN_GRADES
                    return {
                        "kind": "ddd",
                        "heatmap": heatmap,
                        "disc_index": disc_index,
                        "level": DISC_LEVELS[disc_index],
                        "class_index": class_index,
                        "grade": PFRRMANN_GRADES[class_index],
                    }

                # 2) Honest fallback: explain the *localization* of the least
                #    confident landmark.  Never labelled as a grade/DDD CAM.
                conf = np.asarray(
                    decoded.get("localization_conf", []), dtype=float
                ).reshape(-1)
                if conf.size == 0:
                    return None
                keypoint_index = int(np.argmin(conf))
                names = list(_cfg.VERTEBRAE) + list(_cfg.DISC_LEVELS)
                heatmap = cam.cam_for_localization(x, keypoint_index)
                return {
                    "kind": "localization",
                    "heatmap": heatmap,
                    "keypoint_index": keypoint_index,
                    "keypoint": names[keypoint_index],
                    "confidence": float(conf[keypoint_index]),
                    "note": "Attention for landmark localization; not a "
                            "DDD/Pfirrmann grade explanation.",
                }
            finally:
                cam.close()
        except Exception as exc:  # noqa: BLE001 - XAI is optional
            return {"error": str(exc)}

    def extract_features(self, image: Image.Image) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        self._model.eval()
        with torch.no_grad():
            return self._model.extract_features(
                preprocess_image(image).unsqueeze(0).to(self._device)
            )[0].float().cpu()

    def save_annotated(self, image: Image.Image, out_path: Path) -> Path:
        """Save the fully annotated image from the current pipeline run."""
        if not self.available:
            return out_path  # caller handles missing model
        result = self.analyze_full(image)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result["annotated_image"].save(out_path)
        return out_path


def detected_all_landmarks(conf, threshold: float | None = None) -> bool:
    from config import KEYPOINT_CONF_THRESHOLD

    threshold = KEYPOINT_CONF_THRESHOLD if threshold is None else threshold
    conf = np.asarray(conf, dtype=float).reshape(-1)
    return bool(conf.size and conf.min() >= threshold)


def run_vision(image: Image.Image, image_name: str = "",
               engine: Optional[VisionEngine] = None) -> EvidenceCard:
    engine = engine or VisionEngine()
    return engine.analyze(image, image_name)


def _image_size_from_meta(meta: dict):
    """Best-effort checkpoint image size for status reporting."""
    import config as _cfg
    if meta.get("image_size"):
        return int(meta["image_size"])
    return _cfg.IMAGE_SIZE


if __name__ == "__main__":
    engine = VisionEngine()
    print("Vision engine:", engine.status())
