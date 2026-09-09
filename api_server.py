"""
api_server.py
=============

FastAPI REST wrapper around the deterministic decision-support pipeline.
Serves the spine analysis to a frontend.

Endpoints
---------
GET  /api/health
     -> {status, vision: {loaded, checkpoint, error}, gemini_configured}

POST /api/analyze          (multipart/form-data)
     file                   spine image (PNG/JPG)
     use_llm      str       "true"/"false"  (report generation via LLM)
     provider     str       e.g. "gemini"   (LLM provider; "" = default)
     case_id      str       optional case identifier
     clinical     str       JSON  {age, sex, pain_score, pain_duration_years,
                                   symptoms: {...}, image_quality: {...}}
     spondy_predictions str JSON list of spondylolisthesis slips
     -> DecisionSupportResult JSON incl. an `annotated_image_data_url`
        (thin-line anatomy overlay) for direct frontend display.

The vision engine loads lazily from `checkpoints/best_model.pth`. Without a
checkpoint the API stays up: /api/health reports the missing model and
/api/analyze returns a clear 422 (so a frontend can degrade gracefully).

Run locally:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
Run on Colab (see colab_api_run.ipynb): start in a notebook thread and
expose via an ngrok tunnel / cloudflared URL.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image

from decision_support import run_decision_support
import vision_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spine-api")

app = FastAPI(title="Spine MRI Decision Support API", version="1.1.0")

# Frontend-friendly: allow any origin in dev (tighten in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ENGINE: Optional[vision_engine.VisionEngine] = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> vision_engine.VisionEngine:
    """Lazily build the vision engine exactly once (thread-safe)."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = vision_engine.VisionEngine()
    return _ENGINE


def _gemini_configured() -> bool:
    if os.environ.get("GEMINI_API_KEY"):
        return True
    try:
        from config import _load_dotenv
        _load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    return bool(os.environ.get("GEMINI_API_KEY"))


# ------------------------------------------------------------------
# EvidenceCard -> decision-support raw inputs
# ------------------------------------------------------------------

def evidence_to_inputs(card) -> tuple[list, list]:
    """Map an EvidenceCard to decision-support `ddd_predictions`/`geometry`."""
    ddd_predictions = []
    for lf in card.ddd:
        ddd_predictions.append({
            "level": lf.level,
            "pfirrmann_grade": float(lf.pfirrmann_grade),
            "raw_confidence": (
                float(lf.raw_confidence) if lf.raw_confidence is not None else None
            ),
            "calibrated_probability": (
                float(lf.calibrated_probability)
                if lf.calibrated_probability is not None else None
            ),
            "model_version": "vision-engine-v1",
        })

    geometry = []
    for i, g in enumerate(card.geometric_indicators):
        # Disc landmark for level i lives at landmark index 5+i (L1/L2..L4/L5).
        idx = min(len(card.landmark_conf) - 1, 5 + i)
        geometry.append({
            "level": str(g.get("level", "")),
            "offset_ratio": g.get("offset_ratio"),
            "listhesis_possible": g.get("offset_flag"),
            "relative_space": g.get("relative_space"),
            "space_narrowed": g.get("narrowed"),
            "landmark_confidence": round(float(card.landmark_conf[idx]), 3),
        })
    return ddd_predictions, geometry


def _png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------
# Mock vision (demo path — usable before a checkpoint is trained)
# ------------------------------------------------------------------

def mock_evidence_and_draw(image: Image.Image,
                           image_name: str = "") -> tuple:
    """
    Deterministic, clearly-labelled DEMO detections: finds the brightest
    vertical spine column and lays the 10 keypoints along it. Used so the
    whole API (~ thin-line overlay -> evidence -> decision support -> report)
    can be exercised on Colab before `checkpoints/best_model.pth` exists.
    Never to be mistaken for real model output.
    """
    import numpy as np
    from config import DISC_LEVELS
    from schemas import EvidenceCard, LevelFinding
    from utils import draw_landmarks

    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)

    # Brightest column in the central band = rough spine axis.
    cols = gray.mean(axis=0)
    band = slice(int(width * 0.3), int(width * 0.7))
    x_axis = float(int(np.argmax(cols[band]) + band.start))

    ys = height * np.linspace(0.22, 0.72, 5)
    xs = x_axis + np.array([0.0, 4.0, -6.0, 3.0, 8.0])
    verts = np.stack([xs, ys], axis=1)
    discs = np.stack([
        (xs[:-1] + xs[1:]) / 2.0,
        (ys[:-1] + ys[1:]) / 2.0 + height * 0.02,
    ], axis=1)
    points = np.vstack([verts, discs]) / np.array([width, height], dtype=float)
    conf = np.round(np.r_[np.full(5, 0.55), np.full(5, 0.50)], 3)

    distances = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    mean_d = float(distances.mean()) or 1.0
    geo = []
    for i in range(4):
        rel = float(distances[i] / mean_d)
        geo.append({
            "level": DISC_LEVELS[i],
            "relative_space": round(rel, 3),
            "narrowed": rel < 0.85,
            "offset_ratio": round(abs(xs[i] - xs[i + 1]) / mean_d, 3),
            "offset_flag": False,
        })

    ddd = []
    for i, level in enumerate(DISC_LEVELS):
        grade = 2.0 if i in (0, 4) else 3.0      # deterministic, conservative
        ddd.append(LevelFinding(
            level=level, pfirrmann_grade=grade,
            pfirrmann_label="II" if grade == 2 else "III",
            severity="Mild" if grade == 2 else "Moderate",
            class_probabilities=[0.15, 0.35, 0.30, 0.15, 0.05],
            raw_confidence=0.48, calibrated_probability=0.52,
            localization_quality=0.50,
            evidence="MOCK vision demo — not real model output.",
        ))

    card = EvidenceCard(
        image_processed=True, image_name=image_name,
        landmark_points=[[float(x), float(y)] for x, y in points],
        landmark_conf=list(conf),
        geometric_indicators=geo,
        ddd=ddd,
        notes=["MOCK VISION DEMO: synthetic detections until "
               "checkpoints/best_model.pth is trained."],
    )
    annotated = draw_landmarks(
        image, coords=points, confidence=conf)
    return card, annotated


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/api/health")
def health():
    engine = _get_engine()
    return {
        "status": "ok",
        "vision": engine.status(),
        "gemini_configured": _gemini_configured(),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_llm: str = Form("true"),
    provider: str = Form(""),
    case_id: str = Form(""),
    clinical: str = Form("{}"),
    spondy_predictions: str = Form("[]"),
    mock_vision: str = Form(""),
):
    engine = _get_engine()
    mock_mode = (mock_vision.strip().lower() == "true") or (
        mock_vision.strip() == "" and not engine.available
    )

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"detail": f"Bad image: {exc}"})

    if mock_mode:
        card, annotated = mock_evidence_and_draw(image, file.filename or "")
    elif not engine.available:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Vision model is not loaded. Train or upload "
                          "checkpoints/best_model.pth (see colab_training.ipynb), "
                          "or call again with mock_vision=true for the demo path.",
                "model_status": engine.status(),
            },
        )
    else:
        # Vision inference (single forward pass on the CPU threadpool).
        card, annotated = await run_in_threadpool(
            engine.analyze_and_draw, image, file.filename or "")

    try:
        clinical_ctx = json.loads(clinical or "{}")
        spondy_raw = json.loads(spondy_predictions or "[]")
    except json.JSONDecodeError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid JSON field: {exc}"},
        )

    symptoms = dict(clinical_ctx.pop("symptoms", {}) or {})
    image_quality = clinical_ctx.pop("image_quality", None)
    image_name = file.filename or ""

    ddd_predictions, geometry = evidence_to_inputs(card)

    result = run_decision_support(
        case_id=case_id or image_name or "api-case",
        ddd_predictions=ddd_predictions,
        spondy_predictions=spondy_raw,
        geometry=geometry,
        image_quality=image_quality,
        clinical_context=clinical_ctx,
        symptoms=symptoms,
        use_llm=use_llm.strip().lower() == "true",
        provider=(provider or None),
    )

    payload = result.model_dump(mode="json")
    payload["evidence"] = card.model_dump(mode="json")
    payload["annotated_image_data_url"] = _png_data_url(annotated)
    payload["model_status"] = engine.status()
    payload["mock_vision"] = mock_mode
    return payload


# ------------------------------------------------------------------
# Uvicorn entry
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(
        os.environ.get("PORT", "8000")))