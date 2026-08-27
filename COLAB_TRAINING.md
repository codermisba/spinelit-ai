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
| 7        | (Optional) add `ddd_labels.csv` / `spondy_labels.csv` for graded heads |
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
   .venv\Scripts\python.exe cli_pipeline.py --image "dataset/data/processed_tseg_jpgs/case_0000.jpg" --age 58 --sex female --pain-scale 6 --modality mri --pain-years 4 --start-year 2022 --symptoms "low back pain and leg numbness" --pretty
   .venv\Scripts\python.exe app.py      # local Gradio UI
   .venv\Scripts\python.exe evaluate.py # metrics on the validation split
   ```

## Label file formats (optional but recommended for accuracy)

**`ddd_labels.csv`** — DDD grade 0–4 per disc level (Pfirrmann-style):

```csv
filename,level,grade
case_0000.jpg,L1/L2,2
case_0000.jpg,L2/L3,3
```

**`spondy_labels.csv`** — slip percentage (Meyerding derived automatically):

```csv
filename,level,slip_percent
case_0000.jpg,L4/L5,18
```

These labels both train the DDD/spondy heads **and** let
`train_calibrator.py` fit true calibrated probabilities. Without them, the
pipeline uses a documented deterministic fallback calibration.

## Old single-image CLI

The former `predict.py` was replaced by `cli_pipeline.py` (the agentic CLI) —
use `cli_pipeline.py` for single-image jobs now.
