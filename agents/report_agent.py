"""
agents/report_agent.py
======================
Report Writer Agent.

Synthesizes the verified pipeline outputs into a radiologist-style
written report: a structured "findings" section, per-disease diagnoses
with calibrated probabilities and cited evidence, an "impression",
recommendations and an explicit confidence statement + caveats.
"""

from __future__ import annotations

import json

from schemas import (
    ClinicalInterpretation,
    EvidenceCard,
    FinalReport,
    LongitudinalAssessment,
    VerificationResult,
)
from .agent_base import run_json_agent, AgentError

ROLE = (
    "You are a Spine Radiology Report Writer. Produce a clear, professional, "
    "radiologist-style narrative report that is strictly grounded in the "
    "verified evidence provided. Every finding must state its calibrated "
    "probability and cite the evidence supporting it. Explicitly separate "
    "objective imaging findings from symptom-based suggestions. Flag findings "
    "with low confidence. End with an impression and recommendations. "
    "Add a clear statement that this is an AI research aid, not a diagnosis."
)


def _card_compact(card: EvidenceCard | None) -> str:
    if card is None:
        return "No imaging evidence."
    out = []
    if card.ddd:
        out.append("DDD (level: Pfirrmann_grade, label, severity, P, class_probs, conf):")
        for f in card.ddd:
            out.append(
                f"  {f.level}: grade={f.pfirrmann_grade}, "
                f"label={f.pfirrmann_label}, sev={f.severity}, "
                f"class_probs={[round(p,2) for p in f.class_probabilities]}, "
                f"P={f.calibrated_probability}, conf={f.raw_confidence}"
            )
    return "\n".join(out) or "No per-level findings."


def report_writer_agent(
    patient,
    symptoms,
    card: EvidenceCard | None,
    interpretation: ClinicalInterpretation | None,
    verification: VerificationResult | None,
    longitudinal: LongitudinalAssessment | None,
    provider: str | None = None,
) -> FinalReport:
    task = (
        "STRUCTURED SCHEMA (single JSON object):\n"
        "{\n"
        '  "impression": string,\n'
        '  "findings": string (paragraphs),\n'
        '  "diagnoses": [ {"disease": string, "assessment": string, '
        '"calibrated_probability": float, "evidence": [string]} ],\n'
        '  "recommendations": [string],\n'
        '  "confidence_statement": string,\n'
        '  "caveats": [string]\n'
        "}\n\n"
        "PATIENT CLINICAL:\n"
        f"age={patient.age}, sex={patient.sex or 'unspecified'}, "
        f"VAS={patient.pain_scale}, modality={patient.modality or 'unspecified'}, "
        f"pain_years={patient.pain_years}\n\n"
        "PATIENT SYMPTOMS:\n"
        f"{json.dumps(symptoms.model_dump() if symptoms else {}, indent=2)}\n\n"
        "IMAGING FINDINGS (calibrated):\n"
        f"{_card_compact(card)}\n\n"
        "REASONED INTERPRETATION:\n"
        f"{json.dumps(interpretation.model_dump() if interpretation else {}, indent=2)}\n\n"
        "VERIFICATION AUDIT (only include what is confirmed or contested above "
        "threshold):\n"
        f"{json.dumps(verification.model_dump() if verification else {}, indent=2)}\n\n"
        "LONGITUDINAL OUTLOOK:\n"
        f"{json.dumps(longitudinal.model_dump() if longitudinal else {}, indent=2)}"
    )

    try:
        data = run_json_agent(ROLE, task, provider=provider)
        return FinalReport(
            impression=data.get("impression", ""),
            findings=data.get("findings", ""),
            diagnoses=data.get("diagnoses", []),
            recommendations=data.get("recommendations", []),
            confidence_statement=data.get("confidence_statement", ""),
            caveats=data.get("caveats", []),
        )
    except AgentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgentError(f"Invalid report: {exc}") from exc
