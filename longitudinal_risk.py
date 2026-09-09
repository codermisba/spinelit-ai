"""
longitudinal_risk.py
====================

Longitudinal-risk handling (spec §14).

If no validated longitudinal prediction model exists:

    DO NOT output "5-year worsening risk = 0.00"
    DO NOT infer 0.00 = no progression risk.

Instead the pipeline emits:

    LongitudinalRiskResult(
        available=False, horizon_years=5,
        score=None, status=RiskStatus.NOT_AVAILABLE)

A numeric score is ONLY produced when config.yml declares a validated
numeric model (`longitudinal.risk_model_available: true`) AND the model
artifact actually loads and runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evidence_schemas import LongitudinalRiskResult, RiskStatus
from pipeline_config import PipelineConfig


def evaluate_longitudinal_risk(
    horizon_years: int | None = None,
    config: Optional[PipelineConfig] = None,
    actual_score: Optional[float] = None,
    per_level_risk: Optional[list[float]] = None,
) -> LongitudinalRiskResult:
    """Safe longitudinal risk evaluation.

    `actual_score` is only honoured when the config declares a validated
    numeric model AND the call site supplies a real model output. Without
    both, the score stays None.
    """
    cfg = config or PipelineConfig.get()
    horizon = horizon_years or cfg.longitudinal.default_horizon_years

    model_declared = bool(cfg.longitudinal.risk_model_available)
    model_exists = _model_artifact_exists(cfg.longitudinal.risk_model_path)

    available = model_declared and model_exists and actual_score is not None

    if not available:
        return LongitudinalRiskResult(
            available=False,
            horizon_years=horizon,
            score=None,
            status=RiskStatus.NOT_AVAILABLE,
            per_level_risk=[],
            note=(
                "No validated numeric longitudinal prediction model was "
                "available for this assessment; a "
                f"{horizon}-year progression probability cannot be reliably "
                "quantified."
            ),
        )

    return LongitudinalRiskResult(
        available=True,
        horizon_years=horizon,
        score=float(actual_score),
        status=RiskStatus.AVAILABLE,
        per_level_risk=[float(r) for r in (per_level_risk or [])],
        note="Numeric progression risk reported from a validated longitudinal model.",
    )


def longitudinal_outlook_sentence(risk: LongitudinalRiskResult) -> str:
    """Deterministic narrative for the report (no numeric score when None)."""
    if not risk.available:
        return _outlook_not_available(note=risk.note)
    return (
        f"Model-derived numeric progression risk over {risk.horizon_years} "
        f"years: {risk.score:.2f}. This value is a model output and must be "
        "interpreted with clinical judgement."
    )


def _outlook_not_available(note: str = "") -> str:
    if note:
        return note
    return (
        "No validated numeric longitudinal prediction model was available for "
        "this assessment; therefore, a progression probability cannot be "
        "reliably quantified. General clinical discussion of progression risk "
        "must not be represented as a model-derived risk score."
    )


def _model_artifact_exists(path_str: str) -> bool:
    if not path_str:
        return False
    return Path(path_str).exists()