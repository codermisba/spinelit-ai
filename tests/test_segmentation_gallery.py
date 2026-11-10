"""Segmentation gallery + /ui freshness + real-model overlay tests."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api_server
from api_server import app

client = TestClient(app)


def _png_bytes(size=(32, 32), colour=(10, 10, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_segmentation_gallery_lists_and_serves_overlays(tmp_path, monkeypatch):
    folder = tmp_path / "seg"
    folder.mkdir()
    png = _png_bytes()
    (folder / "42_t2_overlay.png").write_bytes(png)
    monkeypatch.setattr(api_server, "SEG_SAMPLE_DIRS", [("test", folder)])

    r = client.get("/api/segmentation/samples")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["samples"][0]["file"] == "42_t2_overlay.png"
    assert body["samples"][0]["source"] == "test"
    assert body["samples"][0]["url"].endswith("/42_t2_overlay.png")

    img = client.get("/api/segmentation/samples/42_t2_overlay.png")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content == png
    assert "no-store" in img.headers["cache-control"]


def test_segmentation_gallery_deduplicates_names(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "x_overlay.png").write_bytes(_png_bytes())
    (second / "x_overlay.png").write_bytes(_png_bytes())

    monkeypatch.setattr(api_server, "SEG_SAMPLE_DIRS",
                        [("samples", first), ("outputs", second)])
    body = client.get("/api/segmentation/samples").json()
    assert body["count"] == 1
    assert body["samples"][0]["source"] == "samples"


def test_segmentation_sample_rejects_unsafe_names():
    for name in ("../secret.png", "a/b.png", "evil.exe", ".hidden.png"):
        resp = api_server.segmentation_sample(name)
        assert resp.status_code in (400, 404)


def test_ui_is_not_cached_and_has_no_demo_mode():
    r = client.get("/ui")
    assert r.status_code == 200
    assert "no-store" in r.headers["cache-control"]
    assert 'id="mock"' not in r.text
    assert "Demo mode" not in r.text
    assert "/api/segmentation/samples" in r.text
    assert 'form.append("mock_vision", "false")' in r.text


def test_committed_gallery_samples_are_real_pngs():
    folder = api_server.CODE_DIR / "examples" / "segmentation_samples"
    files = sorted(folder.glob("*.png"))
    assert files, "expected committed segmentation overlays"
    for path in files:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(
    not (api_server.CODE_DIR / "checkpoints" / "best_model.pth").exists()
    or not (api_server.CODE_DIR / "dataset" / "data" /
            "processed_spider_jpgs" / "100_t2.jpg").exists(),
    reason="trained checkpoint / sample image not available",
)
def test_real_analyze_returns_segmentation_overlay(monkeypatch):
    monkeypatch.delenv("VISION_CHECKPOINT", raising=False)
    monkeypatch.setattr(api_server, "_ENGINE", None)
    image = (api_server.CODE_DIR / "dataset" / "data" /
             "processed_spider_jpgs" / "100_t2.jpg")

    r = client.post(
        "/api/analyze",
        files={"file": ("100_t2.jpg", image.read_bytes(), "image/jpeg")},
        data={"use_llm": "false", "explain": "true", "mock_vision": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mock_vision"] is False
    assert body["ddd_available"] is True
    assert body["segmentation"]["available"] is True
    assert body["segmentation_image_data_url"].startswith(
        "data:image/png;base64,")
    assert body["gradcam_image_data_url"].startswith("data:image/png;base64,")
    assert all(d["confidence"] is not None for d in body["detections"])
