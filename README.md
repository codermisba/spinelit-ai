# Agentic AI Spine Pipeline — DDD

An **end-to-end agentic AI pipeline** for lumbar spine analysis covering
**disc degenerative disease (DDD)** with Pfirrmann grading.

An LLM **orchestrator** coordinates a set of specialist *agents* and a
calibrated *vision engine*. For every case it parses free-text symptoms,
extracts objective imaging findings, fuses them with the clinical record,
**verifies/critiques the interpretation for accuracy**, projects progression,
and writes a radiologist-style report — where every finding carries a
**calibrated probability** and **cited evidence**.

> **Research prototype — not a medical diagnostic tool.**

---

## Architecture (agentic)

```
                 ┌─────────────────────────────────────────────────────┐
 patient inputs  │  Image (X-ray/MRI) · clinical record · free-text    │
                 └────────────────────────┬────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  1 · Symptom Extraction Agent (LLM)                            │
        │      free-text complaints -> StructuredSymptoms                │
        └────────────────────────────────────────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  2 · Vision Engine (agent)  [ConvNeXt + FPN heatmaps]         │
        │      vertebra/disc localization + confidence                   │
        │      DDD Pfirrmann grade (I..V) + calibrated P(DDD)            │
        │      geometric indicators (narrowed space / offset)            │
        │         -> EvidenceCard (machine-readable, cited)              │
        └────────────────────────────────────────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  3 · Clinical–Vision Fusion Reasoning Agent (LLM)              │
        │      clinical + symptoms + EvidenceCard -> ClinicalInterpretation│
        └────────────────────────────────────────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  4 · Verification / Critique Agent (LLM)  ── accuracy loop     │
        │      independent audit; confirms / contests / rejects claims    │
        └────────────────────────────────────────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  5 · Longitudinal Assessment Agent (model + LLM)               │
        │      clinical + image embedding -> future grade + progression   │
        └────────────────────────────────────────────────────────────────┘
                                          ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  6 · Report Writer Agent (LLM)                                 │
        │      radiologist-style report with calibrated probabilities     │
        │      and cited evidence for every claim                        │
        └────────────────────────────────────────────────────────────────┘
                                          ▼
                       Gradio UI · structured tables · written report
```

**Confidence & accuracy guarantees** (the emphasis of this project):

- **Calibrated probabilities (0–1)** per finding — raw model confidence is
  mapped to a real probability of disease (isotonic calibration on labelled
  data; a principled deterministic fallback otherwise). See
  [`calibration.py`](calibration.py) and [`train_calibrator.py`](train_calibrator.py).
- **Per-level numeric confidence** for every finding.
- **Verification / critique loop** — a second, adversarial agent double-checks
  the reasoning agent's claims against the raw evidence and only accepts what
  is confirmed (`agents/verifier_agent.py`).
- **Cited evidence** — the reasoning and report agents must reference the
  specific level + numeric value behind each claim.

### Agents & files

| Agent / module | File | Role |
|---|---|---|
| LLM backend | `llm.py` | Gemini free API / local Ollama client (stdlib only) |
| Schemas | `schemas.py` | shared typed data contracts (each agent in/out) |
| Calibration | `calibration.py` | raw confidence -> calibrated probability |
| Vision engine | `vision_engine.py` | imaging model agent -> `EvidenceCard` |
| Symptom agent | `agents/symptom_agent.py` | free-text -> `StructuredSymptoms` |
| Fusion agent | `agents/fusion_agent.py` | reasoning agent combining evidence |
| Verifier agent | `agents/verifier_agent.py` | critique / accuracy loop |
| Longitudinal agent | `agents/longitudinal_agent.py` | progression outlook |
| Report agent | `agents/report_agent.py` | radiology-style report writer |
| Orchestrator | `orchestrator.py` | coordinates all agents per case |
| UI | `app.py` | Gradio interface |
| CLI | `cli_pipeline.py` | run the pipeline from the command line |
| Imaging model | `model.py`, `train.py`, `dataset.py`, `utils.py` | vision core (see below) |
| Longitudinal model | `clinical_model.py`, `train_clinical.py` | kept progression model |

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
# CPU torch (if not installed):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Set up the LLM (agents)

Get a free key at https://aistudio.google.com/apikey then set:

```bash
# PowerShell
$env:GEMINI_API_KEY="YOUR_KEY"
# or add to your environment so config.py reads it
```

Options in [`config.py`](config.py): `LLM_PROVIDER = "gemini"` (default,
recommended) or `"ollama"` for a fully local model.

---

## Train the vision engine

The imaging model weights stay out of Git (like the rest of the trained
artefacts). The **localizer** is a ConvNeXt + FPN heatmap network with
differentiable soft-argmax (see [`model.py`](model.py)) — it learns *where*
each vertebra/disc is, instead of regressing the population-average position
from a globally pooled vector.

### Checkpoint in this working copy

`checkpoints/best_model.pth` is a `convnext_femto` model trained at **256 px**
on the SPIDER landmark set with the CPU command below. On the held-out
validation split (1690 visible points):

| Metric | Result |
|---|---|
| Mean localization error | **2.35 px** (0.0092 normalized) |
| PCK@5% (≤ 5% image size) | **97.16%** |
| PCK@10% (≤ 10% image size) | **97.93%** |
| Mean landmark confidence | 0.888 |
| DDD exact grade accuracy | **32.9%** teacher-forced / **34.3%** with predicted centres |
| DDD within-one-grade | **79.0%** / **80.0%** |
| DDD grade MAE | **0.96** / **0.92** Pfirrmann grades |
| Segmentation (foreground Dice) | vertebra **0.58**, disc **0.37**, canal **0.51**; mean **0.486** |

DDD numbers are on the 42 held-out SPIDER cases (210 discs). The majority
class scores ≈ 27.5%, so exact-grade accuracy is only modestly above chance:
one labelled sagittal slice per case is genuinely weak evidence for five-level
Pfirrmann grading. Within-one-grade agreement is the more stable/useful
number, and the API also exposes all five class probabilities. This is a
research prototype, **not** a clinical grading tool. Segmentation is the
standalone U-Net (37 held-out cases); its checkpoint metadata stores the same
hard-Dice metric, reproducible with `evaluate_segmentation.py`.

Inspect it directly:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
python preview_predictions.py --num 4       # predicted vs GT side-by-side
```

`preview_predictions.py` writes `outputs/predictions/*_compare.png` (an MRI
with predicted blocks next to the same MRI with ground truth) plus
`summary.txt` with per-point pixel errors, so the blocks can be checked by eye.

### Local CPU run (works on 8 GB RAM, no GPU)

```bash
# same profile as the checkpoint above: 256 px, femto backbone
python train.py --epochs 18 --image-size 256 --batch-size 8 \
    --model convnext_femto --lr 2e-4

# 160 px is ~2.5x faster; both sizes are honoured end-to-end now
python train.py --epochs 18 --image-size 160 --batch-size 8 \
    --model convnext_femto --lr 2e-4

# continue an interrupted run from checkpoints/last_model.pth
python train.py --epochs 18 --image-size 256 --batch-size 8 \
    --model convnext_femto --lr 2e-4 --resume

# quantitative evaluation (px error, PCK@5/10, per-point table)
python evaluate.py --checkpoint checkpoints/best_model.pth
```

`--image-size` is honoured by the data pipeline: `dataset.py` builds its
transform through `utils.get_transform()`, which reads the runtime
`config.IMAGE_SIZE`. This was a real bug (the constant used to be bound at
import time), so training at 160 px while inference used 256 px silently
degraded the API. A regression test in `tests/test_dataset.py` guards it.

Checkpoints carry their own `meta` (`image_size`, `model_name`, `fpn_dim`,
`softargmax_temperature`, `tasks`), so evaluation, the API and Grad-CAM all
preprocess with exactly the training-time configuration.

### Colab GPU (recommended for best accuracy)

1. Open `colab_training.ipynb`, run `prepare_spider.py` on the downloaded
   SPIDER dataset (218 patients, 447 sagittal T2/T2-SPACE series, CC-BY 4.0)
   to build `dataset/coords_pretrain.csv` and write the JPGs, then run all
   cells.
2. Store `best_model.pth` in `checkpoints/` for the pipeline to find it
   (default `checkpoints/best_model.pth`).

### Optional heads

- **DDD / Pfirrmann head** — `python finetune_ddd.py` trains it on cached
  frozen-backbone features whenever `dataset/ddd_labels.csv` exists (fast even
  on CPU) and merges the best head into `checkpoints/best_model.pth` with
  `tasks.ddd=true`. Without labels the API reports `ddd_available=false` and
  shows no grades rather than random ones. `evaluate.py` reports exact
  accuracy, within-one-grade agreement and MAE.
- **Segmentation (U-Net)** — `python prepare_segmentation_pairs.py` builds
  aligned image/mask pairs from SPIDER, `python train_segmentation.py
  --images dataset/segmentation/images --masks dataset/segmentation/masks`
  trains the standalone U-Net, and `python evaluate_segmentation.py`
  reproduces the held-out per-class Dice/IoU with overlay PNGs. Inference uses
  `checkpoints/segmentation.pth` (or downloads it from the Hub). Without masks
  the API reports segmentation as *unavailable*.
- **YOLO detector** — `python yolo_prep.py --from_landmarks` builds
  pseudo-boxes from the landmark CSV, then `python yolo_train.py` writes
  `checkpoints/yolo_best.pt`. Detection priority is
  YOLO → level-labelled landmark boxes → segmentation boxes, always labelled
  with its source.
- **Calibration** — `python train_calibrator.py` fits isotonic calibration for
  **P(Pfirrmann ≥ 2)** on 80% of labelled cases and reports Brier/ECE on the
  held-out 20% (skips cleanly when no trained DDD head exists). On the current
  checkpoint the raw probability is already well calibrated (held-out Brier
  0.111, ECE 0.047) and the isotonic fit was slightly worse (0.119 / 0.070),
  so the artifact keeps the raw probability via a passthrough flag instead of
  shipping a worse correction. Without an artifact the API also serves the
  raw P(Pfirrmann ≥ 2) unchanged (`source="raw-passthrough"`); on a hosted
  deployment `calibration.pkl` is downloaded from the same Hub repo as the
  checkpoint, so no manual copy is needed.

Without a checkpoint the pipeline still runs in a **degraded mode**: it shows
the LLM reasoning/report skeleton and clear model-status, but vision tables
stay empty until the model is trained.

## Hosted API + frontend

The vision engine + decision-support layer are exposed as a REST API that you
can host on **Hugging Face Spaces** or **Render** and call straight from any
frontend over HTTPS:

- `GET  /` and `GET /api/health` — service metadata, model status/version
- `POST /api/analyze` — multipart `file` upload → annotated overlay
  (data-URL PNG), U-Net segmentation overlay, Grad-CAM overlay, per-level DDD
  table, detections with their source, report and model version
- `GET  /api/segmentation/samples` — real held-out segmentation overlays for
  the frontend test gallery
- `GET  /ui` — bundled single-page frontend (`examples/frontend_demo.html`)
- `GET  /docs` — interactive OpenAPI/Swagger UI

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000   # locally
# open http://127.0.0.1:8000/ui
```

The local API uses `checkpoints/best_model.pth` automatically. On a fresh
clone (no local checkpoint) publish the weights to the Hub first and set
`MODEL_ID`, otherwise `/api/analyze` correctly returns **503** instead of
inventing results.

### Where to deploy

| Target | How |
|---|---|
| **Hugging Face Spaces** (free CPU, recommended) | Create a **Docker** Space, push this repo, set `MODEL_ID` (and `HF_TOKEN` if the model repo is private). Public URL: `https://<you>-<space>.hf.space`. |
| **Render** | New **Web Service** → Docker runtime → same env vars. URL: `https://<service>.onrender.com`. |
| **Any VPS / Docker host** | `docker build -t spine-api . && docker run -p 8000:8000 -e MODEL_ID=you/spine-severity spine-api` |

The model is **not baked into the image** (`checkpoints/*.pth` is ignored by
Git and Docker). Publish it once, then every host downloads the active
version at startup:

```bash
python publish_model.py --repo you/spine-severity \
    --with-calibration --with-segmentation
# host env: MODEL_ID=you/spine-severity
```

Both `best_model.pth` and `segmentation.pth` are downloaded from that repo at
startup, so masks and grades work with no files on the host.

See [`docs/DEPLOY_API.md`](docs/DEPLOY_API.md) for the full step-by-step.

Call it from your own frontend:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
     -F "file=@scan.jpg" -F "explain=true" -F "use_llm=false"
```

```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);
fd.append("explain", "true");
fd.append("mock_vision", "false");
const base = "http://127.0.0.1:8000";
const res = await fetch(base + "/api/analyze", {
  method: "POST", body: fd,
});
const data = await res.json();
document.querySelector("#annotated").src = data.annotated_image_data_url;
document.querySelector("#segmentation").src = data.segmentation_image_data_url;
document.querySelector("#gradcam").src = data.gradcam_image_data_url;

// real held-out segmentation overlays for a test gallery
const samples = await (await fetch(`${base}/api/segmentation/samples`)).json();
```

Retraining keeps working without code changes: weights are versioned on the
HF Hub (`weights.py` registry), and each checkpoint carries its own metadata
(`image_size`, `model_name`, `fpn_dim`, `softargmax_temperature`, `tasks`) that
is applied at load time.

- Full deployment guide → [`docs/DEPLOY_API.md`](docs/DEPLOY_API.md)
- Ready-to-use frontend page → [`examples/frontend_demo.html`](examples/frontend_demo.html)
- Publish retrained weights → `python publish_model.py --repo you/spine-severity --with-calibration --with-segmentation`

Unavailable tasks are reported honestly: `/api/analyze` returns
`ddd_available=false` when no Pfirrmann labels were used, segmentation
`available=false` without `checkpoints/segmentation.pth`, and `503` (never
fake results) when no model is configured — unless the caller explicitly
passes `mock_vision=true`.

---

## Run the pipeline

### CLI

```bash
python cli_pipeline.py --image scan.jpg \
    --age 58 --sex female --pain-scale 6 --modality mri \
    --pain-years 4 --start-year 2022 \
    --symptoms "Lower back pain and numbness down my right leg" \
    --years-ahead 5 --pretty
```

Drops `--pretty` to get the full JSON. Saves an annotated image to `outputs/`.

### UI

```bash
python app.py            # http://127.0.0.1:7860
python app.py --share    # public link (Colab)
```

Upload an image, optionally fill the clinical record and free-text symptoms,
and see the annotated image, DDD tables (with calibrated
probabilities + raw confidence), geometric indicators, the verification audit,
the longitudinal outlook and the radiology report.

---

## Deterministic decision-support layer (safety-first)

Alongside the agentic pipeline, this repo ships a **deterministic
decision-support safety layer** (`config.yaml`,
`structured_builder.py`, `contradiction_engine.py`, `evidence_fusion.py`,
`confidence.py`, `longitudinal_risk.py`, `report_validator.py`,
`pipeline_report.py`, `decision_support.py`, `audit_log.py`). It hard-wraps
the model outputs so that:

- **Confidence and evidence are never over-claimed** — raw model confidence is
  classified LOW/MODERATE/HIGH; findings stay `LOW_CONFIDENCE_*`,
  `INDETERMINATE`, `CONTRADICTED` or `NOT_CORROBORATED` until independent
  geometric evidence upgrades them (and even then, "confirmed" requires the
  evidence to explicitly support it).
- **Geometry outranks the model** — explicitly-measured negative geometry
  falsifies a prediction (`CONTRADICTED`, spec §8 R1); positive geometry alone
  cannot validate slip *severity* (Grade III is downgraded to `INDETERMINATE`).
- **No synthetic numbers** — with no validated longitudinal model, risk stays
  `score=None` / `NOTE AVAILABLE`, and the report must disclose that.
- **The LLM can never override the evidence** — it receives only validated
  structured evidence; its output is rejected/regenerated when it violates
  the configured rules and otherwise replaced by a deterministic,
  fully-traceable conservative fallback.
- **Everything is traceable and audited** — every finding carries evidence IDs
  mapped to a registered evidence ledger; runs are logged to `logs/*.jsonl`
  with PHI stripped.

### CLI

```bash
python cli_decision_support.py --example --pretty --no-llm
python cli_decision_support.py --case sample.json --no-llm --out outputs/report.json
```

The existing example shows an all-multilevel case: most model predictions
stay low-confidence or are contradicted by geometry — exactly the honest,
review-gated output the safety layer is designed to produce.

### Tests

```bash
python -m pytest tests/ -q
```

Covers the spec §28 acceptance scenarios (Grade III vs geometry,
contradiction, low-confidence DDD, longitudinal-risk absence, symptom
conservatism, inadequate image quality) plus validator/LLM-fallback and the
drawing overlay tests.

### Annotated-image thin-line overlay

`utils.draw_landmarks(..., anatomy_lines=True)` (default) renders a **thin
1 px schematic** under the landmark markers: vertebra-body outlines, the
spinal-column axis, and horizontal offset lines labelled with the measured
offset ratio. It is derived **only** from the 10 detected keypoints — it is a
landmark-derived schematic, not a learned segmentation mask, and is presented
as such in the legend.

---

## Imaging model (vision core)

- Backbone: **ConvNeXt** (configurable; `convnext_femto` for CPU,
  `convnext_tiny` on GPU) pretrained on ImageNet via `timm`.
- **FPN at stride 4** fuses all backbone stages into a dense feature map.
- **Landmark head** — one Gaussian heatmap per keypoint (L1–L5 + 5 discs);
  coordinates come from a differentiable **soft-argmax** with a sharpening
  temperature (`SOFTARGMAX_TEMPERATURE`) so the expectation reads the actual
  peak instead of collapsing to the image centre.  Per-point confidence is a
  learned head supervised by the model's own localisation error.
- **DDD head** — per disc: Gaussian-attention pooling around the localized
  centre (σ = 4% of the fused map, i.e. roughly one disc, stored in checkpoint
  metadata) **plus** a small CNN over the raw 32×64 image patch, brightness /
  homogeneity statistics, disc spacing and a learned per-level embedding →
  per-level Pfirrmann grade (5-class softmax).  The raw patch matters because
  the fused feature map is trained for landmarks, not disc degeneration.  Only
  trained when `dataset/ddd_labels.csv` exists; otherwise reported as
  unavailable.
- **Segmentation head / U-Net** — optional; a standalone `segmentation.py`
  U-Net (4 classes: background, vertebra, disc, spinal canal) trained with
  Dice+CE (`train_segmentation.py`), loaded from `checkpoints/segmentation.pth`
  or downloaded as `segmentation.pth` from the model Hub repo.  No masks ⇒ no
  fabricated masks.
- **Detector** — optional Ultralytics YOLO (`detector.py`, `yolo_train.py`)
  with landmark/segmentation fallbacks; every box is labelled with its source.
- **Grad-CAM** — hooks the fused FPN feature map (the tensor the DDD head
  consumes).  With a trained DDD head it explains the predicted grade; without
  Pfirrmann labels it returns a **localization** CAM for the least confident
  landmark, tagged `kind="localization"` so it is never presented as a grade
  explanation (`gradcam_demo.py` refuses to imply otherwise).
- Longitudinal `LongitudinalRiskModel` fuses clinical features + image
  embedding to predict future grades and progression risk
  ([`clinical_model.py`](clinical_model.py)).

All preprocessing (resize, ImageNet normalisation) is shared between training,
evaluation, Grad-CAM and the API (`utils.get_transform`), so inference matches
training exactly.

---

## Project status / next steps

- [x] Agent framework (LLM client, schemas, per-agent roles)
- [x] Calibrated probabilities + verification/critique loop
- [x] Orchestrator, CLI, Gradio UI (degraded-mode safe)
- [x] Deterministic decision-support safety layer (evidence fusion,
      contradiction rules, report validation, audit trail, fallback report)
- [x] Thin-line anatomical overlay on annotated images (keypoint-derived)
- [x] Localizer trained on SPIDER landmarks (256 px CPU checkpoint,
      2.35 px mean error / 97.9% PCK@10 on the validation split)
- [x] Test suite for the §28 decision-support acceptance scenarios
- [x] Hosted API (`/api/analyze`, `/ui`, `/docs`) with honest task gating
- [x] DDD/Pfirrmann head trained on SPIDER labels (32.9% exact / 79.0%
      within one grade / 0.96 MAE on held-out discs; see caveat above)
- [x] Standalone U-Net segmentation trained on aligned SPIDER masks
      (mean foreground Dice 0.486; per-class table above)
- [x] Calibration measured on held-out cases (raw P(DDD) Brier 0.111 /
      ECE 0.047; isotonic fit was worse, so the raw probability is served
      via a passthrough artifact)
- [ ] Train the YOLO detector (segmentation + landmark boxes already cover
      detection; YOLO weights are optional)
- [ ] Extend to additional spine diseases / 3D (MRI) inputs
