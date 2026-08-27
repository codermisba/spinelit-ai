"""
agents/symptom_agent.py
=======================
Symptom Extraction Agent.

Parses free-text patient complaints into StructuredSymptoms so the
clinical/vision fusion agent can reason over structured evidence.
Reports its own extraction confidence.
"""

from __future__ import annotations

from schemas import StructuredSymptoms
from .agent_base import run_json_agent, AgentError

ROLE = (
    "You are a Spine Clinical Intake Agent. Your job is to carefully "
    "extract structured clinical symptoms from a patient's own free-text "
    "description. Only record facts the patient actually states; do not "
    "invent findings. Extract anything relevant to lumbar disc "
    "degenerative disease and spondylolisthesis."
)


def symptom_extraction_agent(symptoms_text: str,
                             provider: str | None = None) -> StructuredSymptoms:
    """Parse free-text symptoms into StructuredSymptoms."""
    if not symptoms_text or not symptoms_text.strip():
        return StructuredSymptoms(
            confidence=0.0,
            pain_location="",
            pain_radiation="",
            neuro_symptoms="",
            pain_quality="",
            exacerbating="",
            relieving="",
            pain_duration_hint="",
            red_flags="",
        )

    task = (
        "Structured schema (return these exact fields, all strings):\n"
        "{\n"
        '  "pain_location": string,\n'
        '  "pain_radiation": string,\n'
        '  "neuro_symptoms": string,\n'
        '  "pain_quality": string,\n'
        '  "exacerbating": string,\n'
        '  "relieving": string,\n'
        '  "pain_duration_hint": string,\n'
        '  "red_flags": string,\n'
        '  "confidence": float 0-1 how confident you are the extraction '
        "matches the patient's words\n"
        "}\n\n"
        "Patient's own words (in quotes):\n"
        f'"""{symptoms_text}"""\n\n'
        "If the patient provides no information for a field, use an empty "
        "string for that field. Never fabricate symptoms."
    )

    try:
        data = run_json_agent(ROLE, task, provider=provider)
        return StructuredSymptoms(**data)
    except AgentError:
        raise
    except Exception as exc:  # noqa: BLE001 - tolerate missing fields
        raise AgentError(f"Invalid symptom extraction: {exc}") from exc
