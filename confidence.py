"""
confidence.py
=============

Confidence-status classification (spec §4).

The pipeline MUST NEVER equate:
    raw model confidence  !=  calibrated probability  !=  clinical
    probability          !=  geometric evidence      !=  diagnostic certainty

This module only classifies RAW model confidence into the configured
LOW / MODERATE / HIGH buckets. It never turns a confidence value into a
diagnostic statement.

Raw confidence thresholds are read from config.yaml
(`confidence_thresholds.low` / `.moderate`) and are configurable.
"""

from __future__ import annotations

from typing import Optional

from pipeline_config import PipelineConfig
from evidence_schemas import ConfidenceStatus


def classify_confidence(
    raw_confidence,
    config: Optional[PipelineConfig] = None,
) -> ConfidenceStatus:
    """Classify a raw model confidence (0..1) into a confidence status."""
    if raw_confidence is None:
        # No prediction at all -> not a low-confidence prediction; the
        # caller must handle the absence separately (e.g. NOT_CORROBORATED).
        return ConfidenceStatus.LOW_CONFIDENCE
    cfg = config or PipelineConfig.get()
    low = cfg.confidence.low
    moderate = cfg.confidence.moderate
    rc = float(raw_confidence)
    if rc >= moderate:
        return ConfidenceStatus.HIGH_CONFIDENCE
    if rc >= low:
        return ConfidenceStatus.MODERATE_CONFIDENCE
    return ConfidenceStatus.LOW_CONFIDENCE


def is_low_confidence(
    raw_confidence,
    config: Optional[PipelineConfig] = None,
) -> bool:
    """True when raw confidence is strictly below the low threshold."""
    if raw_confidence is None:
        return False
    cfg = config or PipelineConfig.get()
    return float(raw_confidence) < cfg.confidence.low


def label_for_status(status: ConfidenceStatus) -> str:
    """Human label used inside conservative narrative text."""
    return {
        ConfidenceStatus.HIGH_CONFIDENCE: "high confidence",
        ConfidenceStatus.MODERATE_CONFIDENCE: "moderate confidence",
        ConfidenceStatus.LOW_CONFIDENCE: "low confidence",
    }[status]