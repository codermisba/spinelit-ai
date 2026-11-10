"""
api_server.py
=============

FastAPI REST wrapper around the deterministic decision-support pipeline.
Serves the spine analysis to a frontend over a live public URL (Hugging
Face Spaces / Render / any uvicorn host).

Endpoints
---------
GET  /                  service metadata + docs pointer
GET  /api/health        {status, vision, registry, gemini_configured}
POST /api/analyze       (multipart/form-data) -> AnalyzeResponse

Model registry / retraining
----------------------------
The checkpoint is resolved with `weights.resolve_checkpoint()`:
  VISION_CHECKPOINT = local path or HF Hub model id  (highest priority)
  MODEL_ID          = HF Hub model id                (fallback)
  checkpoints/best_model.pth / release_model.pth     (local fallback)
Every checkpoint carries a `meta` block (image_size, feature stage,
version) written by train.py; `utils.load_model` decodes it so a
retrained model works with zero host-side configuration.  Publish new
weights with `python publish_model.py` and restart the service.

Behaviour without weights (production-safety)
----------------------------------------------
`/api/analyze` NEVER silently invents results.  Without an engine it
returns HTTP 503 + `model_status`.  The clearly-labelled demo path only
runs when callers explicitly pass `mock_vision=true` (or when
ALLOW_MOCK_FALLBACK=true is set AND no model is available).

Environment
-----------
PORT                    uvicorn port (default 8000)
VISION_CHECKPOINT       local path or HF Hub model id for the imaging model
MODEL_ID                HF Hub model id (fallback to VISION_CHECKPOINT)
HF_HUB_REVISION         pinned HF Hub revision (optional)
HF_TOKEN                secret for private model repos
HF_MODEL_FILENAME       file to download from the repo (default best_model.pth)
GEMINI_API_KEY          optional LLM report provider
API_KEYS                comma-separated keys. When set, requests must send
                        `Authorization: Bearer <key>` or `X-Api-Key: <key>`.
CORS_ORIGINS            comma-separated allowed origins (default *).
ALLOW_MOCK_FALLBACK     "true" enables the demo path when no model is loaded.

Run locally:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from PIL import Image

import weights
from decision_support import run_decision_support
from evidence_schemas import DecisionSupportResult
from explanation import build_explanation
import vision_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spine-api")

APP_VERSION = "2.0.0"


# ------------------------------------------------------------------
# Settings from the environment (read once at import for the test path;
# real deployments set these on the host).
# ------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().lower() == "true"


ALLOW_MOCK_FALLBACK = _env_flag("ALLOW_MOCK_FALLBACK", default=False)
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
API_KEYS = [k.strip() for k in os.environ.get("API_KEYS", "").split(",")
            if k.strip()]


app = FastAPI(
    title="Spine MRI Decision Support API",
    description="Lumbar-spine MRI analysis: landmark localization, per-level "
                "Pfirrmann grading, geometry, and a validated decision-support "
                "report. Frontend-ready (multipart image upload) and "
                "model-registry aware.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

_ENGINE: Optional[vision_engine.VisionEngine] = None
_ENGINE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the vision engine at startup (non-fatal on failure)."""
    logger.info("Spine API %s starting. registry=%s",
                APP_VERSION, weights.cache_config_summary())
    try:
        engine = _get_engine()
        logger.info("Vision engine loaded: %s", engine.status())
    except Exception as exc:  # noqa: BLE001 - startup must not crash without weights
        logger.warning("Vision engine failed to load (serving without model): %s", exc)
    yield


app.router.lifespan_context = lifespan


def _get_engine() -> vision_engine.VisionEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            # VisionEngine handles a missing checkpoint internally and sets
            # `.error`; it never raises here, so the service always starts.
            _ENGINE = vision_engine.VisionEngine()
            if not _ENGINE.available:
                logger.warning("No vision model: %s", _ENGINE.error)
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
# Optional API-key gate (API_KEYS env). Frontends send the key in the
# Authorization: Bearer header or X-Api-Key.
# ------------------------------------------------------------------

def _authorize(api_key: Optional[str] = Header(None, alias="X-Api-Key"),
               authorization: Optional[str] = Header(None)):
    if not API_KEYS:
        return True
    supplied = api_key or ""
    if not supplied and authorization:
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
    return bool(supplied) and supplied in API_KEYS


def _require_api_key(api_key: Optional[str] = Header(None, alias="X-Api-Key"),
                     authorization: Optional[str] = Header(None)):
    if _authorize(api_key, authorization):
        return None
    return JSONResponse(
        status_code=401,
        content={"detail": "Unauthorized. Send a valid API key via the "
                           "X-Api-Key header or Authorization: Bearer <key>."},
    )


# ------------------------------------------------------------------
# EvidenceCard -> decision-support raw inputs
# ------------------------------------------------------------------

def evidence_to_inputs(card, model_version: str = "") -> tuple[list, list]:
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
            "model_version": model_version,
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
# Mock vision (explicit demo path — never used implicitly in production)
# ------------------------------------------------------------------

def mock_evidence_and_draw(image: Image.Image,
                           image_name: str = "") -> tuple:
    """
    Deterministic, clearly-labelled DEMO detections: finds the brightest
    vertical spine column and lays the 10 keypoints along it. Used so the
    whole API (~ thin-line overlay -> evidence -> decision support -> report)
    can be exercised without a checkpoint.
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
        ddd_available=True,
        detections=[],
        detection_source="landmark",
        segmentation={"available": False, "source": "none"},
        notes=["MOCK VISION DEMO: synthetic detections."],
    )
    from detector import landmark_detections
    from utils import render_analysis

    detections = landmark_detections(points, width, height)
    card.detections = detections
    annotated = render_analysis(
        image, coords=points, confidence=conf, detections=None,
        show_blocks=True, show_landmarks=True,
    )
    return card, annotated


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

class AnalyzeResponse(DecisionSupportResult):
    """Full /api/analyze payload (decision-support result + evidence + overlay)."""
    evidence: dict = Field(default_factory=dict)
    annotated_image_data_url: str = ""
    gradcam_image_data_url: str = ""
    segmentation_image_data_url: str = ""
    input_image_data_url: str = ""
    explanation: dict = Field(default_factory=dict)
    detections: list[dict] = Field(default_factory=list)
    detection_source: str = ""
    segmentation: dict = Field(default_factory=dict)
    gradcam: dict = Field(default_factory=dict)
    ddd_available: bool = False
    model_status: dict = Field(default_factory=dict)
    mock_vision: bool = False


@app.get("/")
def root():
    engine = _get_engine()
    return {
        "service": "Spine MRI Decision Support API",
        "version": APP_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "ui": "/ui",
        "frontend_integration": "/integration",
        "endpoints": ["/api/health", "/api/analyze", "/ui", "/integration",
                      "/api/segmentation/samples"],
        "vision": engine.status(),
        "registry": weights.cache_config_summary(),
        "gemini_configured": _gemini_configured(),
    }


UI_PATH = Path(__file__).resolve().parent / "examples" / "frontend_demo.html"

# Segmentation prediction galleries.  ``examples/segmentation_samples`` is
# committed so a hosted deployment (which excludes ``outputs/``) still has
# real overlays to show; ``outputs/segmentation`` is picked up automatically
# after running ``evaluate_segmentation.py`` locally.
CODE_DIR = Path(__file__).resolve().parent
INTEGRATION_PATH = CODE_DIR / "docs" / "FRONTEND_INTEGRATION.md"
SEG_SAMPLE_DIRS = [
    ("samples", CODE_DIR / "examples" / "segmentation_samples"),
    ("outputs", CODE_DIR / "outputs" / "segmentation"),
]
_SEG_SAMPLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.png$")


@app.get("/integration", include_in_schema=False)
def integration_doc():
    """Serve the frontend-integration guide (Markdown, never cached)."""
    if not INTEGRATION_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Integration guide not found at {INTEGRATION_PATH}"},
        )
    return FileResponse(
        INTEGRATION_PATH,
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/ui", include_in_schema=False)
def ui():
    """Serve the bundled single-page frontend (never cached)."""
    if not UI_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"Frontend not found at {UI_PATH}"},
        )
    return FileResponse(
        UI_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/segmentation/samples")
def segmentation_samples():
    """List real U-Net overlays available for the frontend gallery."""
    samples, seen = [], set()
    for source, folder in SEG_SAMPLE_DIRS:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.png")):
            if path.name in seen:
                continue
            seen.add(path.name)
            samples.append({
                "name": path.stem,
                "file": path.name,
                "source": source,
                "size_bytes": path.stat().st_size,
                "url": f"/api/segmentation/samples/{path.name}",
            })
    return {
        "count": len(samples),
        "dirs": [str(folder) for _, folder in SEG_SAMPLE_DIRS],
        "samples": samples,
    }


@app.get("/api/segmentation/samples/{filename}", include_in_schema=False)
def segmentation_sample(filename: str):
    """Serve one overlay PNG from the whitelisted gallery directories."""
    if not _SEG_SAMPLE_NAME.match(filename):
        return JSONResponse(
            status_code=400, content={"detail": "Invalid sample name."})
    for _source, folder in SEG_SAMPLE_DIRS:
        try:
            root = folder.resolve()
            candidate = (folder / filename).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate.is_file():
            return FileResponse(
                candidate,
                media_type="image/png",
                headers={"Cache-Control": "no-store, max-age=0"},
            )
    return JSONResponse(
        status_code=404, content={"detail": "Sample not found."})


@app.get("/api/health")
def health():
    engine = _get_engine()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "vision": engine.status(),
        "registry": weights.cache_config_summary(),
        "mock_fallback_allowed": ALLOW_MOCK_FALLBACK,
        "gemini_configured": _gemini_configured(),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    auth_success: dict = Depends(_require_api_key),
    file: UploadFile = File(...),
    use_llm: str = Form("true"),
    provider: str = Form(""),
    case_id: str = Form(""),
    clinical: str = Form("{}"),
    spondy_predictions: str = Form("[]"),
    mock_vision: str = Form(""),
    explain: str = Form("true"),
):
    if isinstance(auth_success, JSONResponse):
        return auth_success

    engine = _get_engine()
    mock_flag = mock_vision.strip().lower() == "true"
    mock_fallback = mock_flag or (mock_vision.strip() == ""
                                  and ALLOW_MOCK_FALLBACK and not engine.available)
    mock_mode = mock_flag if engine.available else mock_fallback

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"detail": f"Bad image: {exc}"})

    gradcam_image = None
    segmentation_image = None
    if mock_mode:
        card, annotated = mock_evidence_and_draw(image, file.filename or "")
        detections = list(card.detections)
        detection_source = card.detection_source
        segmentation = dict(card.segmentation)
        gradcam = {}
        ddd_available = True
    elif not engine.available:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Vision model is not loaded. Configure the model "
                          "registry (MODEL_ID / VISION_CHECKPOINT) or upload "
                          "checkpoints/best_model.pth. For a clearly-labelled "
                          "demo only, call with mock_vision=true.",
                "model_status": engine.status(),
                "registry": weights.cache_config_summary(),
            },
        )
    else:
        # Full vision inference (single forward pass on the threadpool).
        result = await run_in_threadpool(
            engine.analyze_full,
            image,
            file.filename or "",
            explain.strip().lower() == "true",
        )
        card = result["card"]
        annotated = result["annotated_image"]
        detections = result["detections"]
        detection_source = result["detection_source"]
        segmentation = result["segmentation"]
        segmentation_image = result.get("segmentation_image")
        gradcam = result.get("gradcam") or {}
        ddd_available = bool(result["card"].ddd_available)

        heatmap = gradcam.get("heatmap")
        if heatmap is not None:
            try:
                from explainability import overlay_heatmap
                gradcam_image = overlay_heatmap(image, heatmap, alpha=0.55)
            except Exception:  # noqa: BLE001 - optional
                gradcam_image = None

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

    ddd_predictions, geometry = evidence_to_inputs(card, engine.model_version)

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
    payload["gradcam_image_data_url"] = (
        _png_data_url(gradcam_image) if gradcam_image is not None else ""
    )
    payload["segmentation_image_data_url"] = (
        _png_data_url(segmentation_image) if segmentation_image is not None else ""
    )
    payload["input_image_data_url"] = _png_data_url(image)
    payload["detections"] = detections
    payload["detection_source"] = detection_source
    payload["segmentation"] = segmentation
    payload["gradcam"] = {
        k: v for k, v in gradcam.items() if k != "heatmap"
    }
    payload["ddd_available"] = ddd_available
    payload["model_status"] = engine.status()
    payload["mock_vision"] = mock_mode
    payload["explanation"] = build_explanation(
        evidence=payload["evidence"],
        segmentation=segmentation,
        gradcam=gradcam,
        model_status=payload["model_status"],
        use_llm=use_llm.strip().lower() == "true",
        provider=(provider or None),
    )
    return payload


# ------------------------------------------------------------------
# Uvicorn entry
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(
        os.environ.get("PORT", "8000")))