"""Shared fixtures for the decision-support test suite."""

import pytest

from pipeline_config import PipelineConfig
from evidence_schemas import DecisionSupportReport


@pytest.fixture(scope="session")
def cfg():
    return PipelineConfig.get()


class FakeLLMClient:
    """Drop-in for llm.make_client() used by ReportGenerator._call_llm."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, prompt, response_schema=None):
        self.calls += 1
        return self.responses[(self.calls - 1) % len(self.responses)]


@pytest.fixture
def fake_llm_client(monkeypatch):
    def _install(responses):
        import llm
        fake = FakeLLMClient(responses)
        monkeypatch.setattr(llm, "make_client", lambda provider=None: fake)
        return fake
    return _install


def bad_report_payload():
    """A fake LLM payload that must fail ReportValidator."""
    return {
        "overall_status": "NEEDS_REVIEW",
        "objective_imaging_findings": [
            "L1/L2 spondylolisthesis confirmed grade III on model output."
        ],
        "model_confidence": ["low confidence at L4/L5."],
        "contradictions": ["L2/L3 is inconsistent."],
        "provisional_findings": [],
        "confidence_limitations": ["low confidence."],
        "impression": ["L1/L2 confirmed grade III spondylolisthesis."],
        "clinical_correlation": "Correlate clinically.",
        "longitudinal_outlook": "longitudinal risk not available.",
        "disclaimer": "Research prototype. Radiologist review is required.",
        "recommendations": ["Radiologist review."],
    }


def good_report_payload():
    return {
        "overall_status": "NEEDS_REVIEW",
        "objective_imaging_findings": [
            "Automated low-confidence DDD predictions require radiologist "
            "confirmation; longitudinal risk cannot be reliably quantified."
        ],
        "model_confidence": ["L4/L5 DDD raw confidence 0.440 -> LOW_CONFIDENCE."],
        "contradictions": [
            "L2/L3 spondylolisthesis: prediction is inconsistent with "
            "independent geometric analysis (not corroborated)."
        ],
        "provisional_findings": [],
        "confidence_limitations": ["low confidence."],
        "impression": [
            "Automated multilevel findings include low-confidence predictions "
            "that require radiologist confirmation."
        ],
        "clinical_correlation": "Lower-limb symptoms warrant clinical "
                                "neurological correlation.",
        "longitudinal_outlook": "longitudinal risk cannot be reliably "
                                "quantified by the available pipeline output.",
        "disclaimer": "Research prototype. Radiologist review is required.",
        "recommendations": ["Radiologist review."],
        "traceability": [
            {"statement": "s", "evidence_ids": ["E001"]},
        ],
    }