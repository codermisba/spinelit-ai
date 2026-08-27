"""
agents/longitudinal_agent.py
============================
Longitudinal Assessment Agent.

Wraps the LongitudinalRiskModel (clinical + imaging fusion) to predict
future disc-degeneration grade and worsening risk over a horizon, and
asks the LLM to translate the numeric projection into plain-language
progression narrative.
"""

from __future__ import annotations

import json

import torch
from pathlib import Path

from config import LONGITUDINAL_MODEL_NAME, PROGRESSION_HORIZON_YEARS
from schemas import ClinicalInterpretation, EvidenceCard, LongitudinalAssessment
from .agent_base import run_text_agent, AgentError

import clinical_model as clinical
from clinical_model import ClinicalInputs, LongitudinalRiskModel, predict_risk


def _load_risk_model():
    path = Path(LONGITUDINAL_MODEL_NAME)
    if not path.exists():
        return None
    model = LongitudinalRiskModel()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _numeric_prediction(patient, image_features, years_ahead):
    """Run the LongitudinalRiskModel; returns dict or None if untrained."""
    model = _load_risk_model()
    if model is None:
        return None
    if not all(v is not None for v in
               [patient.age, patient.pain_scale, patient.pain_years,
                patient.start_year]):
        return None
    sex = patient.sex or "male"
    modality = (patient.modality or "mri").strip().lower()
    if modality not in {"xray", "mri"}:
        modality = "mri"
    inputs = ClinicalInputs(
        age=int(patient.age), sex=str(sex), pain_scale=float(patient.pain_scale),
        modality=str(modality), pain_years=float(patient.pain_years),
        start_year=int(patient.start_year),
    )
    feats = image_features
    if isinstance(feats, torch.Tensor) and feats.dim() == 1:
        feats = feats
    return predict_risk(
        model, inputs, image_features=feats, years_ahead=years_ahead
    )


def longitudinal_agent(
    patient,
    symptoms,
    card: EvidenceCard | None,
    interpretation: ClinicalInterpretation | None,
    image_features=None,
    years_ahead: float = PROGRESSION_HORIZON_YEARS,
    provider: str | None = None,
) -> LongitudinalAssessment:
    """Assemble the longitudinal assessment (numeric + LLM narrative)."""
    base = LongitudinalAssessment(years_ahead=years_ahead)

    pred = _numeric_prediction(patient, image_features, years_ahead)
    if pred is not None:
        base.overall_risk = round(float(pred["overall_risk"]), 3)
        base.per_level_future_grade = [
            round(float(g), 3) for g in pred["future_grade"]
        ]
        base.per_level_worsening_risk = [
            round(float(r), 3) for r in pred["progression_risk"]
        ]

    # Whether or not the numeric model is trained, ask the LLM to frame
    # the progression outlook narrative given known inputs.
    task = (
        "Write a short, patient-safe, plain-language narrative paragraphs about "
        "the FUTURE progression outlook over "
        f"{years_ahead} years for this patient's lumbar disc degeneration "
        "and spondylolisthesis risk. Base it ONLY on the provided data. If the "
        "numeric longitudinal model was unavailable, say so and reason from "
        "risk factors only. Keep to 1-2 short paragraphs.\n\n"
        "PATIENT CLINICAL:\n"
        f"age={patient.age}, sex={patient.sex or 'unspecified'}, "
        f"VAS pain={patient.pain_scale}, pain_years={patient.pain_years}, "
        f"start_year={patient.start_year}, modality={patient.modality or 'unspecified'}\n\n"
        "NUMERIC LONGITUDINAL OUTPUT (if available):\n"
        f"{json.dumps(pred or {'available': False}, indent=2)}\n\n"
        "CURRENT INTERPRETATION (disease reasoning):\n"
        f"{json.dumps(interpretation.model_dump() if interpretation else {}, indent=2)}"
    )

    try:
        base.narrative = run_text_agent(
            "You are a Spine Longitudinal Outlook Agent.",
            task, provider=provider,
        )
    except AgentError:
        base.narrative = (
            "(Longitudinal narrative unavailable — LLM not configured. "
            f"Numeric outlook risk = {base.overall_risk:.2f} over "
            f"{years_ahead} years.)"
        )

    return base
