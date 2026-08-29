"""
vision_engine.py
================

The *Vision Agent*: wraps the Spine Foundation imaging model
(ConvNeXt-tiny backbone) and turns a single spine image into a
machine-readable `EvidenceCard` — landmarks, geometric indicators and,
per disc level, a Pfirrmann DDD grade with calibrated probability.

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

from config import BEST_MODEL, RELEASE_MODEL
from schemas import EvidenceCard, build_evidence_card
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

        card = build_evidence_card(
            points=decoded["points"],
            localization_conf=decoded["localization_conf"],
            pixels=pixels,
            geo=geo,
            ddd_grade=decoded["ddd_grade"],
            ddd_prob=decoded["ddd_prob"],
            ddd_conf=decoded["ddd_conf"],
            image_name=image_name,
        )
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
