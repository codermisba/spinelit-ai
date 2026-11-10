"""
calibration.py
==============

Turns raw per-level model confidence into *calibrated probabilities*
(0-1) so a reported confidence can be read as a real probability of
disease — the accuracy guarantee the pipeline advertises.

Two modes
---------
1. Fitted calibrators (best). Run `train_calibrator.py` after the DDD head is
   trained. This fits an isotonic regression (scikit-learn) mapping the
   model's raw P(Pfirrmann >= 2) to the observed frequency of degeneration,
   and saves the artifact to CALIB_ARTIFACT. The fit is scored on held-out
   cases (Brier score / ECE) rather than on its own training rows. If the
   fitted map does **not** beat the raw probability on held-out Brier, the
   artifact stores an identity/passthrough flag instead of shipping a worse
   correction.

2. Raw passthrough (no artifact). The head's own P(Pfirrmann >= 2) is served
   unchanged. A hand-tuned heuristic would distort a value that is already a
   softmax probability, so none is applied. ``source="raw-passthrough"``
   makes the absence of fitted calibration explicit.

When ``CALIB_ARTIFACT`` is absent locally but the deployment resolves weights
from a Hub repo (``MODEL_ID``/``VISION_CHECKPOINT``), ``calibration.pkl`` is
downloaded from that repo with the same best-effort semantics as the
standalone segmentation checkpoint.

Localization quality is also folded in: findings from low-confidence
landmarks are down-weighted rather than silently trusted.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import CALIB_ARTIFACT

from schemas import severity_grade as _severity_grade


class _Calibrators:
    """Container for fitted per-task calibrators."""

    def __init__(self):
        self.ddd: Optional[Callable] = None       # raw_conf -> probability
        self.identity: bool = True                # serve the raw probability
        self.source: str = "raw-passthrough"

    def fit_ddd(self, raw_conf, y):
        from sklearn.isotonic import IsotonicRegression
        self.ddd = IsotonicRegression(out_of_bounds="clip")
        self.ddd.fit(np.asarray(raw_conf, dtype=float), np.asarray(y, dtype=float))
        self.identity = False

    def apply_ddd(self, pfirrmann_grade, raw_conf) -> tuple[str, float]:
        """Return (severity label, calibrated probability)."""
        severity = _severity_grade(pfirrmann_grade)
        if self.ddd is not None:
            prob = float(
                np.clip(self.ddd.predict([float(raw_conf)])[0], 0.0, 1.0))
            return severity, prob
        return severity, float(np.clip(raw_conf, 0.0, 1.0))


_calibrators: Optional[_Calibrators] = None


def get_calibrators() -> _Calibrators:
    """Load the shared calibrators once (lazy)."""
    global _calibrators
    if _calibrators is None:
        _calibrators = _load_or_default()
    return _calibrators


def _hub_calibration_path() -> Optional[Path]:
    """Download calibration.pkl from the configured Hub repo, if any."""
    filename = os.environ.get("CALIB_FILENAME", "calibration.pkl")
    try:
        from weights import download_sidecar
        return download_sidecar(filename)
    except Exception:  # noqa: BLE001 - calibration is optional, never fatal
        return None


def _load_or_default() -> _Calibrators:
    path = Path(CALIB_ARTIFACT)
    if not path.exists():
        hub_path = _hub_calibration_path()
        if hub_path is not None:
            path = Path(hub_path)
    if not path.exists():
        return _Calibrators()
    try:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
    except Exception:  # noqa: BLE001 - corrupt artifact -> honest default
        return _Calibrators()
    c = _Calibrators()
    c.ddd = state.get("ddd")
    # An artifact produced from a passthrough decision stores ddd=None; any
    # unreadable/old artifact also degrades to serving the raw probability.
    c.identity = bool(state.get("identity", c.ddd is None))
    c.source = state.get(
        "source", "raw-passthrough" if c.ddd is None else "fitted")
    if c.identity:
        c.ddd = None
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
