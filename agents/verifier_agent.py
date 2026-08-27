"""
agents/verifier_agent.py
========================
Verification / Critique Agent.

Independently audits the fusion agent's `ClinicalInterpretation` against
the raw `EvidenceCard` and the patient's symptoms. Its job is to catch
overstatements, unsupported claims, contradictions and low-confidence
findings — this is the accuracy safety-net of the pipeline (the
"critique loop"). The orchestrator may run it for `VERIFY_ITERATIONS`
passes and only reports a diagnosis as high-confidence when the verifier
confirms it.
"""

from __future__ import annotations

import json

from schemas import (
    ClinicalInterpretation,
    EvidenceCard,
    PatientInput,
    VerificationItem,
    VerificationResult,
)
from .agent_base import run_json_agent, AgentError
from .fusion_agent import _card_to_text

ROLE = (
    "You are a Spine Verification / Critique Agent. A clinical reasoning "
    "agent has produced a draft interpretation for a patient. Your only job "
    "is to independently verify its claims against the objective imaging "
    "evidence and the patient's symptoms. Be strict and adversarial. Mark a "
    "claim as 'rejected' if the evidence contradicts it, 'contested' if the "
    "evidence is weak/ambiguous, 'low_confidence' if the cited imaging "
    "confidence is below 0.5, and 'confirmed' only when the evidence clearly "
    "supports it. Never rubber-stamp. Return your audit as JSON."
)


def verifier_agent(
    patient: PatientInput,
    symptoms,
    card: EvidenceCard | None,
    interpretation: ClinicalInterpretation,
    provider: str | None = None,
) -> VerificationResult:
    task = (
        "STRUCTURED SCHEMA (single JSON object):\n"
        "{\n"
        '  "overall_status": "approved" | "needs_revision" | "rejected",\n'
        '  "items": [\n'
        "      {\"claim\": string, \"supports\": string, \"conflicts\": string, "
        '       "status": "confirmed"|"contested"|"rejected"|"low_confidence", '
        ' "correction": string, "agree_probability": float 0-1}\n'
        "  ],\n"
        '  "summary": string,\n'
        '  "consistency_notes": [string]\n'
        "}\n\n"
        "Provide one item for EACH disease diagnosis and EACH important "
        "claim (including the longitudinal/overall statement if present). "
        "Set agree_probability to how strongly you agree (0-1).\n\n"
        "PATIENT CLINICAL RECORD:\n"
        f"age={patient.age}, sex={patient.sex or 'unspecified'}, "
        f"VAS={patient.pain_scale}, modality={patient.modality or 'unspecified'}, "
        f"pain_years={patient.pain_years}\n\n"
        "PATIENT SYMPTOMS:\n"
        f"{json.dumps(symptoms.model_dump() if symptoms else {}, indent=2)}\n\n"
        "IMAGING EVIDENCE CARD:\n"
        f"{_card_to_text(card)}\n\n"
        "DRAFT INTERPRETATION TO AUDIT:\n"
        f"{json.dumps(interpretation.model_dump(), indent=2)}"
    )

    try:
        data = run_json_agent(ROLE, task, provider=provider)
        items = [VerificationItem(**d) for d in data.get("items", [])]
        overall = data.get("overall_status", "approved")
        if overall not in {"approved", "needs_revision", "rejected"}:
            overall = "approved"
        return VerificationResult(
            overall_status=overall,
            items=items,
            summary=data.get("summary", ""),
            consistency_notes=data.get("consistency_notes", []),
        )
    except AgentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgentError(f"Invalid verification result: {exc}") from exc
