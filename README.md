# Spine Foundation Model

AI-assisted **lumbar spine landmark localization** — a research prototype.

Phase 1 of the Spine Foundation project: given an MRI image, the model predicts
the pixel coordinates of five lumbar levels (L1/L2, L2/L3, L3/L4, L4/L5, L5/S1)
with a confidence value for each level.

> **Research prototype — not a medical diagnostic tool.**
> This software does NOT diagnose disease, predict severity, recommend surgery,
> or provide medical advice. It only performs landmark localization.

---

## Pipeline

```
MRI image
   ↓
Preprocessing (Resize IMAGE_SIZE x IMAGE_SIZE -> ToTensor)
   ↓
Spine Foundation Model (ConvNeXt Tiny backbone)
   ↓
Five lumbar level coordinate predictions (normalized 0-1)
   ↓
Confidence prediction (per level)
   ↓
Visualization (landmarks drawn on ORIGINAL image dimensions)
   ↓
Simple local Gradio UI
```

---

## Repository structure

```
spine-foundation/
│
├── dataset/
│   ├── data/                   # processed images (npy + jpg folders)
│   ├── coords_pretrain.csv     # landmark annotations
│   └── coord_rsna_improved.csv
│
├── checkpoints/                # best_model.pth, latest_model.pth
├── logs/                       # training_log.csv
├── outputs/                    # annotated prediction images
│
├── config.py                   # all paths + hyperparameters
├── dataset.py                  # SpineDataset (image, coords, confidence)
├── model.py                    # SpineFoundationModel (coords, confidence, features)
├── train.py                    # training + validation + checkpoints
├── evaluate.py                 # validation metrics
├── predict.py                  # single-image inference CLI
├── utils.py                    # preprocessing, inference, visualization
├── app.py                      # Gradio UI
└── requirements.txt
```

---

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

On a machine **without** CUDA (recommended for an 8 GB RAM CPU machine),
install the CPU build of PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> The code automatically uses **CUDA when available, otherwise CPU**. No GPU is
> required. The default training resolution is **256x256** (`IMAGE_SIZE` in
> `config.py`) so CPU training stays fast and does not overheat the laptop —
> it is about 4-6x faster and much cooler than 512x512. The batch size is set
> to 4 for an 8 GB RAM machine, and CPU threads are capped
> (`CPU_NUM_THREADS`) to avoid thermal throttling. Set `IMAGE_SIZE = 512` if
> you train on a GPU and want maximum accuracy.

---

## Commands

### 1. Sanity check the dataset

```bash
python dataset.py
```

Expected output: the number of images in `coords_pretrain.csv`, followed by the
shape of one sample (`image` → `Tensor(3,512,512)`, `coords` → `Tensor(10)`,
`confidence` → `Tensor(5)`).

### 2. Sanity check the model

```bash
python model.py
```

Expected output: forward-pass shapes for a random batch
(`coords` → `Tensor(B,10)`, `confidence` → `Tensor(B,5)`,
`features` → `Tensor(B,512)`).

### 3. Train the model

```bash
python train.py
```

Additional options:

```bash
python train.py --epochs 20          # override epoch count
python train.py --resume             # resume from checkpoints/latest_model.pth
python train.py --max-batches 5      # quick smoke test (few batches/epoch)
```

Expected output:

- A train/validation split summary (≈80% / 20% of the images).
- Per-epoch progress bars for train and validation with running loss and LR.
- `logs/training_log.csv` with one row per epoch
  (`Epoch, Train Loss, Validation Loss, Learning Rate, Timestamp`).
- `checkpoints/best_model.pth` (lowest validation loss) and
  `checkpoints/latest_model.pth` (every epoch).
- Early stopping after 10 epochs without validation improvement.
- Training resumes automatically from `latest_model.pth` when present.

### 4. Evaluate the model

```bash
python evaluate.py
# or: python evaluate.py --checkpoint checkpoints/latest_model.pth
```

Expected output (valid only after training has produced a checkpoint):

```
## Foundation Model Evaluation

Validation Samples : 225

Coordinate MAE (0-1) : ...
Coordinate MSE (0-1) : ...

Mean Localization Error (px) : ...

Per-Level Localization Error (px):
L1/L2 : ...
L2/L3 : ...
...

Confidence Statistics:
L1/L2 : ...
...
```

If no checkpoint exists, the script prints a message and exits gracefully.

### 5. Test one MRI image

```bash
python predict.py --image "dataset/data/processed_tseg_jpgs/case_0000.jpg"
```

Expected output: the device and checkpoint used, the image resolution, and for
each level:

```
L1/L2
X           : 268.3
Y           : 318.1
Confidence  : 0.982
```

An annotated image (points + level labels + confidence values, plus ground
truth if the filename is in the annotation CSV) is saved to
`outputs/<name>_<timestamp>_annotated.jpg`. The original image is never
modified. Use `--output` to choose a custom save path.

### 6. Launch the local UI

```bash
python app.py
```

Expected output: a local URL, by default:

```
Running on local URL:  http://127.0.0.1:7860
```

Open it in a browser. The UI shows:

- Title: **Spine Foundation Model**
- Description: *AI-assisted lumbar spine landmark localization*
- MRI image upload + **Analyze** button
- Annotated output image
- Prediction table (`Level | X | Y | Confidence`, rows L1/L2 … L5/S1)
- **Model Status:** Loaded / Not Loaded
- **Checkpoint:** best_model.pth
- Disclaimer: *Research prototype — not a medical diagnostic tool.*

If `checkpoints/best_model.pth` is missing, the UI still launches and shows
"No trained model checkpoint found. Please train the model first." — it does
not crash.

---

## Model architecture

- **Backbone:** pretrained ConvNeXt Tiny (`timm`, `num_classes=0`)
- **Shared head:** BatchNorm → Dropout → Linear(→512) → GELU → **512-d features**
- **Coordinate head:** predicts 10 values (x,y for 5 levels), sigmoid → 0-1
- **Confidence head:** predicts 5 values, sigmoid → 0-1

## Training details

- Loss: `total_loss = coordinate_mse_loss + 0.1 * confidence_bce_loss`
- **Confidence supervision:** the dataset labels all confidences as 1, so the
  head is trained (self-supervised) to predict the model's own per-level
  localization error instead: `confidence_target = exp(-err / 0.05)`. This
  gives real per-image confidence at inference time
  (`CONFIDENCE_SUPERVISION` in `config.py`).
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Scheduler: CosineAnnealingLR
- Gradient clipping: max norm 1.0
- Split: stratified by dataset source, 80/20
- Early stopping patience: 10 epochs

---

## Next research phases (NOT implemented yet)

1. Foundation localization (this phase)
2. Lumbar / disc region extraction
3. DDD dataset
4. DDD classification
5. Severity classification
6. Explainability
7. Longitudinal Spine Intelligence
