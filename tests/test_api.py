"""
API server tests.

No trained checkpoint is required: without one the health endpoint reports
the missing model and /api/analyze returns a clear 422 so the frontend can
degrade gracefully.
"""

import base64

from fastapi.testclient import TestClient
from PIL import Image

from api_server import app, evidence_to_inputs, _png_data_url
from schemas import EvidenceCard, LevelFinding

client = TestClient(app)


def test_health_reports_status_ok():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "vision" in body
    assert isinstance(body["gemini_configured"], bool)


def test_analyze_without_checkpoint_returns_422_when_mock_disabled():
    r = client.post(
        "/api/analyze",
        files={"file": ("scan.png", _png_bytes(), "image/png")},
        data={"use_llm": "true", "mock_vision": "false"},
    )
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body
    assert body.get("model_status") is not None


def test_analyze_bad_image_returns_400():
    r = client.post(
        "/api/analyze",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_evidence_to_inputs_mapping():
    card = EvidenceCard(
        image_processed=True,
        landmark_points=[[0.5, 0.4] for _ in range(10)],
        landmark_conf=[0.9] * 10,
        geometric_indicators=[
            {"level": "L1/L2", "relative_space": 1.02, "narrowed": False,
             "offset_ratio": 0.04, "offset_flag": False},
            {"level": "L4/L5", "relative_space": 0.72, "narrowed": True,
             "offset_ratio": 0.05, "offset_flag": False},
        ],
        ddd=[
            LevelFinding(level="L1/L2", pfirrmann_grade=3.0, raw_confidence=0.44,
                         calibrated_probability=0.45),
            LevelFinding(level="L5/S1", pfirrmann_grade=2.0, raw_confidence=0.6,
                         calibrated_probability=0.55),
        ],
    )
    ddd, geo = evidence_to_inputs(card)
    assert len(ddd) == 2
    assert ddd[0]["level"] == "L1/L2"
    assert geo[0]["level"] == "L1/L2"
    # narrowed -> space_narrowed True; offset_flag -> listhesis_possible
    assert geo[1]["space_narrowed"] is True
    assert geo[0]["listhesis_possible"] is False
    assert "landmark_confidence" in geo[0]


def test_png_data_url_shape():
    url = _png_data_url(Image.new("RGB", (8, 8)))
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_mock_vision_end_to_end():
    """Without a checkpoint the demo path still returns a full result."""
    r = client.post(
        "/api/analyze",
        files={"file": ("scan.png", _png_bytes(), "image/png")},
        data={
            "use_llm": "false",
            "mock_vision": "true",
            "clinical": '{"age": 52, "symptoms": {"right_leg_numbness": true}}',
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mock_vision"] is True
    assert body["report"] is not None
    assert body["report"]["overall_status"]
    assert body["annotated_image_data_url"].startswith("data:image/png;base64,")
    assert body["evidence"]["image_processed"] is True
    assert body["case"]["overall_status"] == "NEEDS_REVIEW"


def _png_bytes() -> bytes:
    import io
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 10, 30)).save(buf, format="PNG")
    return buf.getvalue()