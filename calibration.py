"""
calibration.py
==============

Turns raw per-level model confidence into *calibrated probabilities*
(0-1) so a reported confidence can be read as a real probability of
disease — the accuracy guarantee the pipeline advertises.

Two modes
---------
1. Fitted calibrators (best). On Colab, run `train_calibrator.py` with
   labelled Pfirrmann (DDD) data. This fits an isotonic
   regression (scikit-learn) mapping raw confidence -> observed
   probability, and saves the artifact to CALIB_ARTIFACT.

2. Deterministic fallback (no labels). A principled heuristic so the
   pipeline is fully self-contained on a fresh clone:
       P  = 0.6 * severity_prior + 0.4 * raw_model_confidence
   This is clearly documented as an uncalibrated fallback.

Localization quality is also folded in: findings from low-confidence
landmarks are down-weighted rather than silently trusted.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import CALIB_ARTIFACT

from schemas import calibrate_ddd as _fallback_ddd


class _Calibrators:
    """Container for fitted per-task calibrators."""

    def __init__(self):
        self.ddd: Optional[Callable] = None       # raw_conf -> probability
        self.source: str = "deterministic-fallback"

    def fit_ddd(self, raw_conf, y):
        from sklearn.isotonic import IsotonicRegression
        self.ddd = IsotonicRegression(out_of_bounds="clip")
        self.ddd.fit(np.asarray(raw_conf, dtype=float), np.asarray(y, dtype=float))

    def apply_ddd(self, pfirrmann_grade, raw_conf) -> tuple[str, float]:
        """Return (severity label, calibrated probability)."""
        severity, fallback_prob = _fallback_ddd(pfirrmann_grade, raw_conf)
        if self.ddd is not None:
            prob = float(np.clip(self.ddd.predict([float(raw_conf)])[0], 0.0, 1.0))
            return severity, prob
        return severity, fallback_prob


_calibrators: Optional[_Calibrators] = None


def get_calibrators() -> _Calibrators:
    """Load the shared calibrators once (lazy)."""
    global _calibrators
    if _calibrators is None:
        _calibrators = _load_or_default()
    return _calibrators


def _load_or_default() -> _Calibrators:
    c = _Calibrators()
    path = Path(CALIB_ARTIFACT)
    if path.exists():
        try:
            with open(path, "rb") as fh:
                state = pickle.load(fh)
            c.ddd = state.get("ddd")
            c.source = state.get("source", "fitted")
        except Exception:  # noqa: BLE001 - corrupt artifact -> default
            c = _Calibrators()
    return c


def calibrate_ddd(pfirrmann_grade: float, raw_conf: float) -> tuple[str, float]:
    return get_calibrators().apply_ddd(pfirrmann_grade, raw_conf)


def adjustment_factor(localization_quality: float) -> float:
    """Down-weight calibrations from uncertain landmarks (smoothing)."""
    q = float(np.clip(localization_quality, 0.0, 1.0))
    return 0.5 + 0.5 * q     # range [0.5, 1.0]


if __name__ == "__main__":
    c = get_calibrators()
    print("Calibration source:", c.source)
    print("DDD  grade=3 conf=0.9 ->", c.apply_ddd(3.0, 0.9))
