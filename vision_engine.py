"""
vision_engine.py
================

The *Vision Agent*: wraps the Spine Foundation imaging model
(ConvNeXt-tiny backbone) and turns a single spine image into a
machine-readable `EvidenceCard` — landmarks, geometric indicators and,
per disc level, DDD grade + spondylolisthesis slip with calibrated
probabilities.

This is the numeric "eyes" of the pipeline. Everything it produces is
later cited by the LLM reasoning and reporter agents, so accuracy here
directly bounds overall accuracy.

If no trained checkpoint is available the vision engine reports this
clearly in `model_status` (localizer weights are trained on Colab per
`colab_training.ipynb`); the pipeline can drop to a degraded mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from calibration import calibrate_ddd, calibrate_spondy
from config import BEST_MODEL, CONFIDENCE_THRESHOLD, RELEASE_MODEL
from schemas import EvidenceCard, LevelFinding
from utils import (
    compute_geometric_indicators,
    decode_outputs,
    load_model,
    norm_to_pixels,
    predict,
    preprocess_image,
)


class VisionEngine:
    """Runs the imaging model and assembles an EvidenceCard."""

    def __init__(self, checkpoint: Optional[str] = None):
        self._loaded = False
        self._model = None
        self._device = None
        self._checkpoint_name = None
        self.error = ""
        self.load(checkpoint or str(BEST_MODEL))

    def load(self, checkpoint: str):
        try:
            self._model, self._device, self._checkpoint_name = load_model(
                checkpoint
            )
            self._loaded = True
            self.error = ""
        except FileNotFoundError as exc:
            self._loaded = False
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def checkpoint_name(self) -> str:
        return str(self._checkpoint_name) if self._checkpoint_name else "none"

    def status(self) -> dict:
        return {"loaded": self.available, "checkpoint": self.checkpoint_name,
                "error": self.error}

    @torch.no_grad()
    def _run_model(self, image: Image.Image) -> dict:
        outputs = predict(self._model, preprocess_image(image), self._device)
        return decode_outputs(outputs)

    def analyze(self, image: Image.Image, image_name: str = "") -> EvidenceCard:
        if not self.available:
            card = EvidenceCard(
                image_processed=False, image_name=image_name,
                notes=[self.error or "Vision model not loaded."],
            )
            return card

        decoded = self._run_model(image)
        width, height = image.size
        pixels = norm_to_pixels(decoded["points"], width, height)
        geo = compute_geometric_indicators(pixels)

        card = EvidenceCard(image_processed=True, image_name=image_name)
        card.landmark_points = [[float(a), float(b)] for a, b in pixels]
        card.landmark_conf = [float(c) for c in decoded["localization_conf"]]
        card.geometric_indicators = geo

        from config import DISC_LEVELS

        for i, level in enumerate(DISC_LEVELS):
            # ---- DDD ----
            d_grade = float(decoded["ddd_grade"][i])
            d_rc = float(decoded["ddd_conf"][i])
            d_sev, d_prob = calibrate_ddd(d_grade, d_rc)
            d_ev = (
                f"DDD grade {d_grade:.2f}/4 ({d_sev}); "
                f"calibrated probability {d_prob:.2f}; "
                f"raw model confidence {d_rc:.2f}."
            )
            card.ddd.append(LevelFinding(
                level=level, grade=round(d_grade, 3), severity=d_sev,
                raw_confidence=round(d_rc, 3),
                calibrated_probability=round(d_prob, 3),
                localization_quality=round(float(decoded["localization_conf"][i]), 3),
                evidence=d_ev,
            ))

            # ---- Spondylolisthesis ----
            s_slip = float(decoded["spondy_slip_pct"][i])
            s_rc = float(decoded["spondy_conf"][i])
            s_grade, s_prob = calibrate_spondy(s_slip, s_rc)
            s_ev = (
                f"Slip {s_slip:.1f}% ({s_grade}); "
                f"calibrated probability {s_prob:.2f}; "
                f"raw model confidence {s_rc:.2f}."
            )
            card.spondy.append(LevelFinding(
                level=level, slip_percent=round(s_slip, 1), meyerding=s_grade,
                raw_confidence=round(s_rc, 3),
                calibrated_probability=round(s_prob, 3),
                localization_quality=round(float(decoded["localization_conf"][i]), 3),
                evidence=s_ev,
            ))

        return card

    def extract_features(self, image: Image.Image) -> Optional[torch.Tensor]:
        if not self.available:
            return None
        self._model.eval()
        with torch.no_grad():
            return self._model.extract_features(
                preprocess_image(image).unsqueeze(0).to(self._device)
            )[0].float().cpu()

    def save_annotated(self, image: Image.Image, out_path: Path) -> Path:
        """Save an annotated image from the current pipeline run."""
        if not self.available:
            return out_path  # caller handles missing model
        decoded = self._run_model(image)
        from utils import draw_landmarks
        annotated = draw_landmarks(
            image, coords=decoded["points"],
            confidence=decoded["localization_conf"],
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(out_path)
        return out_path


def run_vision(image: Image.Image, image_name: str = "",
               engine: Optional[VisionEngine] = None) -> EvidenceCard:
    engine = engine or VisionEngine()
    return engine.analyze(image, image_name)


if __name__ == "__main__":
    engine = VisionEngine()
    print("Vision engine:", engine.status())
