"""
agents/fusion_agent.py
======================
Clinical–Vision Fusion Reasoning Agent.

Takes the structured patient record, the parsed symptoms and the vision
engine's `EvidenceCard`, and produces a reasoned `ClinicalInterpretation`
for Disc Degenerative Disease (DDD, Pfirrmann grading) — citing the
specific evidence behind each differential and reporting uncertainties.
"""

from __future__ import annotations

import json

from schemas import (
    ClinicalInterpretation,
    Differential,
    EvidenceCard,
    PatientInput,
)
from .agent_base import run_json_agent, AgentError

ROLE = (
    "You are a Spine Clinical Reasoning Agent. You integrate a patient's "
    "clinical record, their structured symptoms and the objective numeric "
    "findings from an imaging model (the Evidence Card) to reason about "
    "Disc Degenerative Disease (DDD) using the Pfirrmann grading (grades "
    "I-V, reported per disc level). "
    "Ground every conclusion in the cited evidence; clearly separate "
    "what is supported by imaging vs. suggested by symptoms. Where evidence "
    "is absent or contradicts a diagnosis, say so and lower the likelihood. "
    "Be conservative: never claim a definitive grade without adequate "
    "confidence and geometric support — mark low-confidence or "
    "indeterminate findings as such. "
    "This is a research aid, not a diagnosis."
)


def _card_to_text(card: EvidenceCard | None) -> str:
    if card is None:
        return "No imaging evidence available."
    lines = []
    lines.append(f"image_processed: {card.image_processed}")
    if card.landmark_conf:
        lines.append(
            "landmark_conf (L1-L5 vert then 5 discs): "
            + ", ".join(f"{c:.2f}" for c in card.landmark_conf)
        )
    if card.geometric_indicators:
        lines.append("geometric_indicators:")
        for g in card.geometric_indicators:
            lines.append(
                f"  - {g.get('level')} relative_space={g.get('relative_space')} "
                f"space_narrowed={g.get('narrowed')} "
                f"offset_ratio={g.get('offset_ratio')} "
                f"listhesis_possible={g.get('offset_flag')}"
            )
    if card.ddd:
        lines.append("DDD (Pfirrmann) per level:")
        for f in card.ddd:
            lines.append(f"  - {f.level}: {f.model_dump()}")
    if card.notes:
        lines.append("notes: " + "; ".join(card.notes))
    return "\n".join(lines)


def _clinical_to_text(patient: PatientInput) -> str:
    return (
        f"age={patient.age}, sex={patient.sex or 'unspecified'}, "
        f"VAS pain={patient.pain_scale}, modality={patient.modality or 'unspecified'}, "
        f"pain_years={patient.pain_years}, start_year={patient.start_year}"
    )


def fusion_agent(patient: PatientInput, symptoms, card: EvidenceCard | None,
                 provider: str | None = None) -> ClinicalInterpretation:
    task = (
        "STRUCTURED SCHEMA to return (a single JSON object):\n"
        "{\n"
        '  "summaries": { "<disease_key>": narrative string, ... },  '
        "keys: disc_degenerative_disease\n"
        '  "differentials": [\n'
        "      {\"disease\": string, \"likelihood\": float 0-1, "
        '"rationale": string, "evidence_ids": [string]}\n'
        "  ],\n"
        '  "key_evidence": [string],\n'
        '  "uncertainties": [string],\n'
        '  "recommendations": [string]\n'
        "}\n\n"
        "Evidence IDs you may cite: 'clinical-record', 'symptoms', "
        "and per-level imaging citations such as 'dd:L1/L2', 'geo:L3/L4' "
        "(from the Evidence Card level labels). "
        "You must reference the actual numeric values.\n\n"
        "PATIENT CLINICAL RECORD (from 'clinical-record'):\n"
        f"{_clinical_to_text(patient)}\n\n"
        "PATIENT SYMPTOMS (from 'symptoms'):\n"
        f"{json.dumps(symptoms.model_dump() if symptoms else {}, indent=2)}\n\n"
        "IMAGING EVIDENCE (from the Evidence Card):\n"
        f"{_card_to_text(card)}"
    )

    try:
        data = run_json_agent(ROLE, task, provider=provider)
        diff = [Differential(**d) for d in data.get("differentials", [])]
        return ClinicalInterpretation(
            summaries=data.get("summaries", {}),
            differentials=diff,
            key_evidence=data.get("key_evidence", []),
            uncertainties=data.get("uncertainties", []),
            recommendations=data.get("recommendations", []),
        )
    except AgentError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AgentError(f"Invalid fusion interpretation: {exc}") from exc
