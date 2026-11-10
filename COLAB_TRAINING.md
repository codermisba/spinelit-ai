# Training the Vision Engine on Colab

The repository ships **code only** — no weights are committed. You train the
imaging model (the pipeline's *vision engine*) on a free Colab GPU, keep the
checkpoints in Google Drive, then download them for local testing.

The **agentic LLM layer** (symptom agent, reasoning/fusion agent,
verification/critique agent, longitudinal agent, report writer) already works
without training — it calls Gemini via your `GEMINI_API_KEY`. This notebook
only gives the pipeline its **eyes**.

## Quick path

1. Open `colab_training.ipynb` in Google Colab (**File → Upload notebook**).
2. Set your repo URL in cell 2 (and optionally `GEMINI_KEY` to enable LLM
   agents inside Colab).
3. Make sure your dataset is in Drive (default: `MyDrive/dataset/`).
4. Run the cells top-to-bottom.

## CPU-only profile (8 GB RAM, no GPU)

The same `train.py` runs on CPU. The checkpoint in this working copy
(`checkpoints/best_model.pth`) was produced with:

```powershell
& .\.venv\Scripts\python.exe train.py --epochs 18 --image-size 256 `
    --batch-size 8 --model convnext_femto --lr 2e-4
& .\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/best_model.pth
```

It reaches **2.35 px mean error / 97.9% PCK@10** on the validation split (see
the evaluation table). `--image-size 160` is ~2.5x faster and is honoured
end-to-end; the checkpoint stores its own `image_size` / `model_name` /
`softargmax_temperature`, so evaluation, Grad-CAM and the API automatically
reproduce the training-time preprocessing.

`convnext_femto` @256px is roughly 3–4 s/batch on 2 threads; `convnext_tiny`
@256px is much slower on CPU and is best left to Colab.

## What each notebook section does

| Section | What it does |
|---------|--------------|
| 0–4      | Set repo URL, check GPU, mount Drive, clone repo, install deps |
| 5–6      | Load dataset from Drive (folder or `dataset.zip`), sanity checks |
| 7        | Run `prepare_spider.py` on SPIDER to build coords + `ddd_labels.csv` |
| 8        | Switch config to GPU values (`IMAGE_SIZE=512`, `BATCH_SIZE=32`, `convnext_tiny`) |
| 9–10     | **Train** (`python train.py --model convnext_tiny --epochs 60 --image-size 256 --batch-size 32`) and evaluate |
| 11       | (Optional) fit **calibrated probabilities** (`train_calibrator.py`; skips without a trained DDD head) |
| 12       | **Test the trained model through the agentic CLI** (`cli_pipeline.py`) |
| 13       | (Optional) launch the Gradio agentic UI with a public `--share` link |
| 14       | **Save checkpoints to Google Drive** (survives runtime reset) |
| 15       | Download `best_model.pth` / `calibration.pkl` to your PC |
| 16       | Restore checkpoints from Drive later |

## DDD head (Pfirrmann) — train separately from the localizer

The localizer backbone is frozen once it is good; the DDD head is then trained
on cached features (one forward pass per case), which is ~30–40 s/epoch on CPU
instead of hours.

```bash
# after prepare_spider.py produced dataset/ddd_labels.csv
python finetune_ddd.py --epochs 250 --patience 40 --select-by loss
```

It merges the best head into `checkpoints/best_model.pth`, sets
`tasks.ddd=true` in the checkpoint metadata and records the validation metrics
(`ddd_finetune` in `meta`, including `best_epoch` and `val_within_one`).
Selection is by validation cross-entropy by default (`--select-by loss`);
`--select-by within_one` is available but noisy on ~210 validation discs.
`evaluate.py --checkpoint checkpoints/best_model.pth` reproduces the
teacher-forced metrics and reports within-one-grade agreement too.

Optional segmentation / YOLO heads:

```bash
# standalone U-Net on an aligned image/mask folder (or --nnunet_dir)
python prepare_segmentation_pairs.py        # build dataset/segmentation/
python train_segmentation.py --images dataset/segmentation/images \
    --masks dataset/segmentation/masks --epochs 40
python evaluate_segmentation.py             # held-out Dice/IoU + overlays

# YOLO from landmark-derived pseudo-boxes (no extra annotation needed)
python yolo_prep.py --from_landmarks
python yolo_train.py --epochs 60
```

These write `checkpoints/segmentation.pth` and `checkpoints/yolo_best.pt`.
When either file is absent the API reports that task as unavailable instead of
inventing masks/boxes; detection falls back to landmark-derived blocks and the
response field `detection_source` says which source was used. The segmentation
trainer and `evaluate_segmentation.py` share the same case-level 20% split
(seed 42), so the reported Dice matches the training-time value.

## Save the model for later use

Cell 14 copies to `MyDrive/spine-checkpoints/`:

- `best_model.pth` — the imaging model (all tasks the vision engine uses)
- `last_model.pth` — resume point
- `segmentation.pth` — standalone U-Net masks (optional)
- `calibration.pkl` — fitted calibrated probabilities
- `longitudinal_model.pth` — progression model (if you train `train_clinical.py`)

Publish everything the hosted API needs in one command:

```bash
python publish_model.py --repo acme/spine-severity \
    --with-calibration --with-segmentation
```

The API downloads `best_model.pth`, `segmentation.pth` and
`calibration.pkl` from the Hub repo named by `MODEL_ID`; nothing has to be
copied onto the host.

## Use the trained model back home (Windows)

1. Run notebook **section 15** → download `best_model.pth` (and
   `calibration.pkl`).
2. Put them in your local repo:
   ```
   E:\spine-foundation\checkpoints\best_model.pth
   E:\spine-foundation\checkpoints\calibration.pkl
   ```
3. Test:
   ```powershell
   .venv\Scripts\python.exe evaluate.py --checkpoint checkpoints\best_model.pth
   .venv\Scripts\python.exe preview_predictions.py --num 4
   .venv\Scripts\python.exe gradcam_demo.py --image "dataset/data/processed_spider_jpgs/case_midsag.jpg"
   .venv\Scripts\python.exe cli_pipeline.py --image "dataset/data/processed_spider_jpgs/case_midsag.jpg" --age 58 --sex female --pain-scale 6 --modality mri --pain-years 4 --start-year 2022 --symptoms "low back pain and leg numbness" --pretty
   .venv\Scripts\python.exe app.py      # local Gradio UI
   .venv\Scripts\uvicorn.exe api_server:app --port 8000   # REST API + /ui frontend
   ```
   `preview_predictions.py` writes side-by-side predicted-vs-ground-truth
   images to `outputs/predictions/` for visual QA. `gradcam_demo.py` refuses
   to explain an untrained DDD head (use `--allow-untrained` only for an
   architecture demo).

## Label file formats (produced by the SPIDER prep script)

Run `prepare_spider.py` on the downloaded **SPIDER** dataset (218 patients,
447 sagittal T2/T2-SPACE series, CC-BY 4.0). It selects the midsagittal T2
slice + segmentation, derives the vertebral/disc centroids and the DDD labels,
and writes the JPGs to `dataset/data/processed_spider_jpgs/`:

```bash
python prepare_spider.py --data_dir /path/to/SPIDER_data
```

It writes two CSVs:

**`dataset/coords_pretrain.csv`** — disc landmark coordinates (used for
localization; also `dataset/coords_vertebrae.csv` optional, else vertebra
centres are derived automatically via `DERIVE_VERTEBRA_CENTERS`):

```csv
filename,level,relative_x,relative_y
case_midsag.jpg,L1/L2,0.42,0.33
```

**`dataset/ddd_labels.csv`** — Pfirrmann grade 1–5 per disc level:

```csv
filename,level,pfirrmann_grade
case_midsag.jpg,L3/L4,3
```

These labels both train the multi-class DDD Pfirrmann head **and** let
`train_calibrator.py` fit true calibrated probabilities. Without
`ddd_labels.csv` the DDD head is left untrained: the API returns
`ddd_available=false`, calibration is skipped, and no random grades are shown.

## Localization metric

`train.py`/`evaluate.py` report **PCK@5% / PCK@10%** (fraction of landmarks
within 5% / 10% of the image size) plus per-point pixel error. The checkpoint
in this working copy (femto, 256 px) scores **2.35 px mean error, 97.16%
PCK@5 and 97.93% PCK@10** on 1690 visible validation points. The soft-argmax
temperature in `config.py` is what makes those coordinates come from the
heatmap peak rather than the image centre. Run `preview_predictions.py` to
see the predicted blocks next to ground truth.

## Old single-image CLI

The former `predict.py` was replaced by `cli_pipeline.py` (the agentic CLI) —
use `cli_pipeline.py` for single-image jobs now.
