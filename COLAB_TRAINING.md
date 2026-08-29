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

## What each notebook section does

| Section | What it does |
|---------|--------------|
| 0–4      | Set repo URL, check GPU, mount Drive, clone repo, install deps |
| 5–6      | Load dataset from Drive (folder or `dataset.zip`), sanity checks |
| 7        | Run `prepare_spider.py` on SPIDER to build coords + `ddd_labels.csv` |
| 8        | Switch config to GPU values (`IMAGE_SIZE=512`, `BATCH_SIZE=32`) |
| 9–10     | **Train** (`python train.py --epochs 60`) and evaluate |
| 11       | (Optional) fit **calibrated probabilities** (`train_calibrator.py`) |
| 12       | **Test the trained model through the agentic CLI** (`cli_pipeline.py`) |
| 13       | (Optional) launch the Gradio agentic UI with a public `--share` link |
| 14       | **Save checkpoints to Google Drive** (survives runtime reset) |
| 15       | Download `best_model.pth` / `calibration.pkl` to your PC |
| 16       | Restore checkpoints from Drive later |

## Save the model for later use

Cell 14 copies to `MyDrive/spine-checkpoints/`:

- `best_model.pth` — the imaging model (all tasks the vision engine uses)
- `last_model.pth` — resume point
- `calibration.pkl` — fitted calibrated probabilities
- `longitudinal_model.pth` — progression model (if you train `train_clinical.py`)

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
   .venv\Scripts\python.exe cli_pipeline.py --image "dataset/data/processed_spider_jpgs/case_midsag.jpg" --age 58 --sex female --pain-scale 6 --modality mri --pain-years 4 --start-year 2022 --symptoms "low back pain and leg numbness" --pretty
   .venv\Scripts\python.exe app.py      # local Gradio UI
   .venv\Scripts\python.exe evaluate.py # metrics on the validation split
   ```

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
`train_calibrator.py` fit true calibrated probabilities. Without them, the
pipeline uses a documented deterministic fallback calibration.

## Old single-image CLI

The former `predict.py` was replaced by `cli_pipeline.py` (the agentic CLI) —
use `cli_pipeline.py` for single-image jobs now.
