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
        │  2 · Vision Engine (agent)  [ConvNeXt-tiny via timm]           │
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

## Train the vision engine (Colab GPU, recommended)

The imaging model has no weights committed (kept out of Git like the robot
norm). Train it on Colab:

1. Open `colab_training.ipynb`, run `prepare_spider.py` on the downloaded
   SPIDER dataset (218 patients, 447 sagittal T2/T2-SPACE series, CC-BY 4.0)
   to build `dataset/coords_pretrain.csv` and `dataset/ddd_labels.csv` and
   write the JPGs, then run all cells.
2. Store `best_model.pth` in `checkpoints/` for the pipeline to find it
   (default `checkpoints/best_model.pth`).

For **real calibrated probabilities**, provide grading labels and run:

```bash
python train_calibrator.py   # fits isotonic calibration from labelled data
```

Without a checkpoint the pipeline still runs in a **degraded mode**: it shows
the LLM reasoning/report skeleton and clear model-status, but vision tables
stay empty until the model is trained.

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

## Imaging model (vision core)

- Backbone: **ConvNeXt-Tiny** pretrained on ImageNet via `timm`, shared 512-d
  embedding feeding multiple heads (see [`model.py`](model.py)):
  - coordinate head → L1–L5 + disc centres, with per-point self-supervised
    confidence
  - DDD head → per-level Pfirrmann grade (multi-class softmax, cross-entropy)
- Longitudinal `LongitudinalRiskModel` fuses clinical features + image
  embedding to predict future grades and progression risk
  ([`clinical_model.py`](clinical_model.py)).

The pipeline's accuracy is bounded by the vision engine, so training the
localizer on a good labelled set (and fitting the calibrator) is the priority
for a strong final-year result.

---

## Project status / next steps

- [x] Agent framework (LLM client, schemas, per-agent roles)
- [x] Calibrated probabilities + verification/critique loop
- [x] Orchestrator, CLI, Gradio UI (degraded-mode safe)
- [ ] Train localizer + DDD head on labelled data (Colab)
- [ ] Fit calibration from labels; report calibration curves / AUROC
- [ ] Extend to additional spine diseases / 3D (MRI) inputs
