# Spine Foundation Model

AI-assisted **lumbar spine analysis from X-ray / MRI** — a research prototype.

Given a sagittal lumbar image the model:

1. **Localizes all lumbar vertebrae (L1–L5) and disc levels (L1/L2 … L5/S1)**
   with a per-point **confidence score**.
2. Estimates **disc degeneration (DDD)** grade (0–4) per level with confidence.
3. Estimates **spondylolisthesis** slip % + Meyerding grade per level with
   confidence.
4. Computes label-free **geometric indicators** (narrowed disc space,
   centre-offset → possible listhesis).
5. Fuses **clinical inputs** — age, sex, pain scale (VAS), imaging modality
   (X-ray/MRI), years in pain, symptom start year — into a **longitudinal risk
   model** that predicts future grades and progression probability for any
   horizon (1–10 years).

> **Research prototype — not a medical diagnostic tool.**
> DDD / spondylolisthesis heads remain untrained (and are flagged as such)
> until labelled data is provided.

---

## Prebuilt models

No off-the-shelf model matches this exact pipeline (closest: TotalSpineSeg =
nnU-Net MRI segmentation; RSNA-style models = classification only). We therefore
build on the standard foundation: an **ImageNet-pretrained ConvNeXt-Tiny
backbone via `timm`**, with task-specific heads trained on our landmark CSVs.
Vertebra-centre supervision is derived automatically from disc centroids when
explicit vertebra labels are absent.

## Pipeline

```
X-ray / MRI image ──► preprocessing (resize 256², tensor)
        │
        ▼
ConvNeXt-Tiny backbone (ImageNet-pretrained, timm)
        │  512-d shared embedding
        ├─ coordinate head          → 10 points (5 vertebrae + 5 discs), 0-1
        ├─ localization confidence  → per-point self-supervised confidence
        ├─ DDD head                 → grade 0-4 + confidence   (needs labels)
        └─ spondylosis head         → slip % + confidence      (needs labels)

patient record (age, sex, VAS pain, modality, pain years, start year)
        │
        ▼
LongitudinalRiskModel (+ image embedding, + horizon in years)
        └─ predicted future grades · progression risk · overall risk
```

---

## Repository structure

```
spine-foundation/
├── dataset/
│   ├── coords_pretrain.csv       # disc landmarks (committed)
│   ├── coords_rsna_improved.csv  # extra annotations (committed)
│   ├── ddd_labels.csv            # optional — add to train DDD grading
│   ├── spondy_labels.csv         # optional — add to train slip estimation
│   ├── longitudinal_records.csv  # optional — clinical records
│   └── data/                     # images (gitignored; upload to Drive)
├── colab_training.ipynb          # one-click Colab training notebook
├── COLAB_TRAINING.md             # detailed Colab guide
├── config.py                     # all paths + hyperparameters
├── dataset.py                    # multi-task dataset (masks, derived vertebrae)
├── model.py                      # SpineFoundationModel (all heads)
├── clinical_model.py             # LongitudinalRiskModel + ClinicalInputs
├── train.py                      # multi-task imaging training
├── train_clinical.py             # longitudinal risk model training
├── evaluate.py                   # validation metrics
├── predict.py                    # single-image CLI report
├── export_release.py             # portable fp16 artifact (NOT committed)
├── app.py                        # Gradio UI (image + clinical inputs)
└── requirements.txt
```

**No trained weights are committed.** Train on Colab and store checkpoints in
Google Drive — see `COLAB_TRAINING.md`.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # CPU: first run
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Train

Locally (CPU, slow): `python train.py`
On **Colab GPU** (recommended): open `colab_training.ipynb` and run all cells.

The trainer auto-detects which tasks have labels:

| Labels present | Tasks trained |
|---|---|
| `coords_pretrain.csv` only | landmarks + confidence |
| + `ddd_labels.csv` | + DDD grading |
| + `spondy_labels.csv` | + spondylolisthesis |
| + `longitudinal_records.csv` | longitudinal risk (`python train_clinical.py`) |

## Predict

```bash
python predict.py --image scan.jpg
python predict.py --image scan.jpg --age 58 --sex female --pain-scale 6 \
    --modality mri --pain-years 4 --start-year 2022 --years-ahead 5
```

Prints every vertebra/disc with pixel position + confidence, geometric
indicators, DDD grades, Meyerding slips, and (with clinical inputs) the
longitudinal progression report. Saves an annotated image to `outputs/`.

## UI

```bash
python app.py            # http://127.0.0.1:7860
python app.py --share    # public link (Colab)
```

Upload an image, optionally fill patient data, get tables for landmarks,
geometry, DDD and spondylolisthesis — all with confidence scores.

## Training details

- Loss: masked coordinate MSE + self-supervised confidence BCE
  (`conf_target = exp(-error/temperature)`) + SmoothL1 grade/slip losses when
  labelled
- AdamW (1e-4) · CosineAnnealingLR · grad-clip 1.0 · AMP on GPU
- Stratified 80/20 split · early stopping (patience 10) · resume support
- Checkpoints: `checkpoints/best_model.pth`, `last_model.pth`,
  `longitudinal_model.pth`

---

## Next phases

1. ~~Foundation localization~~ ✅ (this phase)
2. Annotate DDD + spondylolisthesis labels (Pfirrmann / Meyerding)
3. Retrain with labels → calibrated grading
4. Explainability (Grad-CAM per level)
5. Prospective longitudinal validation
