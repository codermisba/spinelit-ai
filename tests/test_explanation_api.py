"""Explanation payload + integration-guide tests.

The explanation must always be grounded in real model evidence and never
invent findings.  Without an LLM key it is deterministic; with an LLM it is
replaced only when the call succeeds, otherwise it falls back and records
the error.
"""

import pytest
from fastapi.testclient import TestClient

import llm
from api_server import app
from explanation import build_explanation, build_facts, deterministic_text

client = TestClient(app)


def _evidence():
    return {
        "image_size": [256, 256],
        "detections": [{"label": "L1", "box": [1, 2, 3, 4]}],
        "detection_source": "landmark",
        "landmark_points": [[0.5, 0.4] for _ in range(10)],
        "landmark_conf": [0.9, 0.8, 0.7, 0.95, 0.85],
        "ddd_available": True,
        "ddd": [
            {
                "level": "L1/2",
                "pfirrmann_grade": 3.0,
                "pfirrmann_label": "III",
                "severity": "Moderate",
                "calibrated_probability": 0.64,
                "localization_quality": 0.82,
            },
            {
                "level": "L2/3",
                "pfirrmann_grade": 4.0,
                "pfirrmann_label": "IV",
                "severity": "Severe",
                "calibrated_probability": 0.93,
                "localization_quality": 0.96,
            },
            {
                "level": "L3/4",
                "pfirrmann_grade": 2.0,
                "pfirrmann_label": "II",
                "severity": "Mild",
                "calibrated_probability": 0.31,
                "localization_quality": 0.91,
            },
        ],
    }


def test_deterministic_explanation_reports_severity_and_grade():
    facts = build_facts(
        _evidence(),
        segmentation={"available": True, "source": "unet", "boxes": [{}, {}]},
        gradcam={"kind": "ddd", "level": "L2/3", "grade": 4},
        model_status={"model_version": "v1", "segmentation": {"val_dice": 0.486}},
    )
    # Most severe level is listed first.
    assert facts["levels"][0]["level"] == "L2/3"

    text = deterministic_text(facts)
    assert "L2/3" in text
    assert "grade 4" in text
    assert "severe" in text.lower()
    assert "P(DDD) 0.93" in text
    assert "most severe" in text.lower()
    assert "radiologist" in text.lower()
    # Segmentation + Grad-CAM are described only because they were provided.
    assert "U-Net segmentation" in text
    assert "Grad-CAM" in text


def test_deterministic_explanation_marks_unavailable_ddd():
    evidence = _evidence()
    evidence["ddd_available"] = False
    evidence["ddd"] = []
    text = deterministic_text(build_facts(evidence))
    assert "Pfirrmann grading is unavailable" in text
    assert "no per-level degeneration grade" in text


def test_build_explanation_skips_llm_when_disabled():
    out = build_explanation(_evidence(), use_llm=False)
    assert out["source"] == "deterministic"
    assert out["llm_used"] is False
    assert out["error"] == ""
    assert "Pfirrmann" in out["text"] or "grade" in out["text"]


def test_build_explanation_uses_llm_when_available(monkeypatch):
    class FakeClient:
        model = "fake-model"

        def complete_text(self, prompt):
            assert "L2/3" in prompt  # facts are passed to the model
            return "The model found severe degeneration at L2/3."

    monkeypatch.setattr(llm, "make_client", lambda provider=None: FakeClient())
    out = build_explanation(_evidence(), use_llm=True, provider="gemini")
    assert out["source"] == "gemini"
    assert out["llm_used"] is True
    assert out["model"] == "fake-model"
    assert out["text"].startswith("The model found severe")


def test_build_explanation_falls_back_when_llm_fails(monkeypatch):
    def boom(provider=None):
        raise RuntimeError("no api key")

    monkeypatch.setattr(llm, "make_client", boom)
    out = build_explanation(_evidence(), use_llm=True, provider="gemini")
    assert out["source"] == "deterministic"
    assert out["llm_used"] is False
    assert "no api key" in out["error"]
    assert "most severe" in out["text"].lower()


def test_build_explanation_falls_back_on_empty_llm_output(monkeypatch):
    class EmptyClient:
        model = "empty"

        def complete_text(self, prompt):
            return "   "

    monkeypatch.setattr(llm, "make_client", lambda provider=None: EmptyClient())
    out = build_explanation(_evidence(), use_llm=True)
    assert out["source"] == "deterministic"
    assert "empty" in out["error"].lower()


def test_integration_guide_is_served_as_markdown():
    r = client.get("/integration")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "no-store" in r.headers["cache-control"]
    body = r.text
    assert "Frontend Integration Guide" in body
    assert "/api/analyze" in body
    assert "mock_vision" in body
    assert "explanation" in body
    assert "/api/segmentation/samples" in body


def test_docker_context_includes_integration_guide():
    """A hosted image without the guide would serve a 404 at /integration."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    patterns = [
        line.strip()
        for line in (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # `docs/` or `docs` would drop the guide before the image is built.
    assert "docs/" not in patterns
    assert "docs" not in patterns
    assert (root / "docs" / "FRONTEND_INTEGRATION.md").is_file()


def test_analyze_returns_input_image_and_explanation(monkeypatch):
    """The demo path still exercises the real explanation plumbing."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 10, 30)).save(buf, format="PNG")

    r = client.post(
        "/api/analyze",
        files={"file": ("scan.png", buf.getvalue(), "image/png")},
        data={"use_llm": "false", "mock_vision": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["input_image_data_url"].startswith("data:image/png;base64,")
    explanation = body["explanation"]
    assert explanation["text"]
    assert explanation["source"] == "deterministic"
    assert explanation["llm_used"] is False
    # No fabricated severe findings on the synthetic demo.
    assert "radiologist" in explanation["text"].lower()
