"""
test_agents.py
==============

Offline validation of the agent contracts WITHOUT calling a real LLM.

A FakeClient impersonates the Gemini/Ollama client and returns canned
JSON, so we can confirm that:
  - each agent's JSON schema parses,
  - the pydantic models validate,
  - the orchestrator wires all agents together end to end.

Run:  python test_agents.py
"""

from __future__ import annotations

from types import SimpleNamespace

import agents.agent_base as ab
from agents import (
    symptom_extraction_agent,
    fusion_agent,
    verifier_agent,
    longitudinal_agent,
    report_writer_agent,
)
from orchestrator import AgenticPipeline
from schemas import ClinicalInterpretation, EvidenceCard, PatientInput
from vision_engine import VisionEngine
from calibration import calibrate_ddd, calibrate_spondy

RESULTS: dict[str, str] = {}


class _FakeLLM:
    temperature = 0.2

    def complete_text(self, prompt):
        # Route by the role marker inside the prompt.
        if "Clinical Intake Agent" in prompt:
            key = "symptoms"
        elif "Clinical Reasoning Agent" in prompt:
            key = "fusion"
        elif "Verification / Critique Agent" in prompt:
            key = "verifier"
        elif "Longitudinal Outlook Agent" in prompt:
            key = "longitudinal"
        elif "Radiology Report Writer" in prompt:
            key = "report"
        else:
            key = "unknown"
        return RESULTS.get(key, "{}")


def _install_fake():
    # Temporarily replace make_client in agent_base with a fake factory.
    fake = _FakeLLM()
    orig = ab.make_client
    ab.make_client = lambda provider=None: fake

    def restore():
        ab.make_client = orig

    return restore


def _fake_evidence() -> EvidenceCard:
    card = EvidenceCard(image_processed=True, image_name="case.jpg")
    points = [[i * 10, 100 + i * 5] for i in range(10)]
    card.landmark_points = points
    card.landmark_conf = [0.9] * 10
    card.geometric_indicators = [
        {"level": "L1/L2", "relative_space": 1.0, "narrowed": False,
         "offset_ratio": 0.1, "offset_flag": False},
        {"level": "L2/L3", "relative_space": 0.7, "narrowed": True,
         "offset_ratio": 0.12, "offset_flag": False},
        {"level": "L3/L4", "relative_space": 0.6, "narrowed": True,
         "offset_ratio": 0.3, "offset_flag": True},
    ]
    from schemas import LevelFinding
    for level, grade, slip in zip(
            ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"],
            [1.0, 2.2, 3.1, 2.0, 1.5],
            [2.0, 8.0, 20.0, 5.0, 3.0]):
        sev, p = calibrate_ddd(grade, 0.8)
        mg, sp = calibrate_spondy(slip, 0.7)
        card.ddd.append(LevelFinding(
            level=level, grade=grade, severity=sev,
            calibrated_probability=p, raw_confidence=0.8,
            localization_quality=0.9, evidence=f"grade {grade}"))
        card.spondy.append(LevelFinding(
            level=level, slip_percent=slip, meyerding=mg,
            calibrated_probability=sp, raw_confidence=0.7,
            localization_quality=0.9, evidence=f"slip {slip}"))
    return card


def main():
    restore = _install_fake()
    RESULTS.update({
        "symptoms": ('{"pain_location":"lower back","pain_radiation":"leg",'
                     '"neuro_symptoms":"numbness","pain_quality":"mechanical",'
                     '"exacerbating":"sitting","relieving":"walking",'
                     '"pain_duration_hint":"2 years","red_flags":"",'
                     '"confidence":0.9}'),
        "fusion": ('{"summaries":{'
                   '"disc_degenerative_disease":"moderate DDD at L3/L4",'
                   '"spondylolisthesis":"mild slip at L3/L4"},'
                   '"differentials":[{"disease":"disc_degenerative_disease",'
                   '"likelihood":0.8,"rationale":"narrowed space",'
                   '"evidence_ids":["dd:L3/L4","geo:L3/L4"]}],'
                   '"key_evidence":["dd:L3/L4"],"uncertainties":["none"],'
                   '"recommendations":["imaging follow-up"]}'),
        "verifier": ('{"overall_status":"approved",'
                     '"items":[{"claim":"DDD at L3/L4","supports":"grade 3.1",'
                     '"conflicts":"","status":"confirmed",'
                     '"agree_probability":0.9}],'
                     '"summary":"consistent","consistency_notes":["ok"]}'),
        "longitudinal": ("Plain-language narrative over 5 years."),
        "report": ('{"impression":"moderate DDD","findings":"Findings text.",'
                   '"diagnoses":[{"disease":"disc_degenerative_disease",'
                   '"assessment":"moderate at L3/L4","calibrated_probability":0.8,'
                   '"evidence":["dd:L3/L4"]}],'
                   '"recommendations":["follow-up"],'
                   '"confidence_statement":"moderate",'
                   '"caveats":["research aid"]}'),
    })

    patient = PatientInput(
        age=58, sex="female", pain_scale=6, modality="mri",
        pain_years=4, start_year=2022,
        symptoms_text="low back pain radiating to leg",
    )
    card = _fake_evidence()

    s = symptom_extraction_agent(patient.symptoms_text)
    assert s.pain_location == "lower back" and s.confidence == 0.9, s
    print("symptom agent OK")

    interp = fusion_agent(patient, s, card)
    assert isinstance(interp, ClinicalInterpretation)
    assert len(interp.differentials) == 1
    print("fusion agent OK")

    verify = verifier_agent(patient, s, card, interp)
    assert verify.overall_status == "approved" and len(verify.items) == 1
    print("verifier agent OK")

    lo = longitudinal_agent(patient, s, card, interp,
                            image_features=None, years_ahead=5)
    assert "5 years" in lo.narrative
    print("longitudinal agent OK")

    rep = report_writer_agent(patient, s, card, interp, verify, lo)
    assert rep.impression == "moderate DDD" and len(rep.diagnoses) == 1
    print("report agent OK")

    # Full orchestrator with a fake vision engine (so it has evidence).
    class FakeVision(VisionEngine):
        def __init__(self):
            self._loaded = True

        @property
        def available(self):
            return True

        @property
        def checkpoint_name(self):
            return "fake"

        def status(self):
            return {"loaded": True, "checkpoint": "fake"}

        def analyze(self, image, image_name=""):
            return _fake_evidence()

        def extract_features(self, image):
            import torch
            return torch.zeros(512)

        def save_annotated(self, image, out_path):
            from pathlib import Path
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            image.save(p)
            return p

    from PIL import Image
    import tempfile, os
    dummy = Image.new("RGB", (64, 64), (128, 128, 128))
    result = AgenticPipeline(vision_engine=FakeVision()).run(
        patient, pil_image=dummy)
    assert result.evidence and result.evidence.image_processed
    assert result.interpretation and result.interpretation.differentials
    assert result.verification and result.verification.overall_status == "approved"
    assert result.longitudinal and result.longitudinal.narrative
    assert result.report and result.report.impression
    assert not result.errors, result.errors
    print("orchestrator end-to-end (fake LLM + fake vision) OK")

    restore()
    print("\nALL AGENT TESTS PASSED")


if __name__ == "__main__":
    main()
