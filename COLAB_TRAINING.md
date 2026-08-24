# Training on Google Colab

The repository ships **code only** — no weights are committed. Train on a free
Colab GPU, then keep the checkpoints in Google Drive.

## Quick path

Open `colab_training.ipynb` in Colab (**File → Upload notebook**) and run the
cells top to bottom. The sections below explain every step.

## Step-by-step

### 1. Put the dataset in Google Drive

You already have a dataset folder in Drive — that works as-is. By default the
notebook expects it at:

```
MyDrive/dataset/
```

If it is somewhere else (e.g. `MyDrive/spine-foundation/dataset`), just edit
the `DATASET_DIR` line in **cell 5** of the notebook:

```python
DATASET_DIR = '/content/drive/MyDrive/your-folder-name'
```

The folder should contain (a zip named `dataset.zip` works too):

```
data/processed_lsd_jpgs/     # image folders (same layout as local repo)
data/processed_osf_jpgs/
data/processed_spider_jpgs/
data/processed_tseg_jpgs/
coords_pretrain.csv          # disc landmarks  — already in the repo, but include if you have updates
ddd_labels.csv               # DDD grades      (optional)
spondy_labels.csv            # slip %          (optional)
longitudinal_records.csv     # clinical records (optional)
```

The folder contents are merged into `dataset/` in Colab, so any label CSVs
you add to Drive automatically override the repo defaults.

### 2. Open Colab

1. Go to <https://colab.research.google.com>
2. **Runtime → Change runtime type → GPU (T4)**
3. Upload / open `colab_training.ipynb`

### 3. Run the notebook cells

| Cell | What it does |
|------|--------------|
| 1-2 | Checks GPU, mounts Drive |
| 3   | Clones this repo |
| 4   | Installs `timm`, `gradio`, etc. |
| 5   | Unzips `dataset.zip` into `dataset/` |
| 6   | Switches config to GPU values (`IMAGE_SIZE=512`, `BATCH_SIZE=32`) |
| 7   | Dataset + model sanity checks |
| 8   | **Trains** (`python train.py --epochs 60`) |
| 9   | Evaluates (per-point error + confidence) |
| 10  | Single-image prediction demo |
| 11  | Trains the longitudinal risk model (if records CSV present) |
| 12  | **Copies all checkpoints to `MyDrive/spine-foundation/checkpoints/`** |
| 13  | Launches the Gradio UI with a public `--share` link |

### 4. Save the model for further use

Cell 12 copies everything to Drive:

- `checkpoints/best_model.pth` — imaging model (all tasks)
- `checkpoints/last_model.pth` — resume point
- `checkpoints/longitudinal_model.pth` — clinical risk model
- optionally `checkpoints/release_model.pth` via
  `python export_release.py --output /content/drive/...` (portable fp16)

### 5. Use the trained model back home

Copy the checkpoint from Drive into your local `checkpoints/` folder, then:

```bash
python predict.py --image scan.jpg                       # full report
python app.py                                            # local UI
python evaluate.py                                       # metrics
```

## Label file formats

**`ddd_labels.csv`** — per disc level, grade 0-4 (Pfirrmann-style):

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

**`longitudinal_records.csv`** — one row per patient visit/timepoint:

```csv
age,sex,pain_scale,modality,pain_years,start_year,years_ahead,baseline_grade_L1/L2,...,future_grade_L5/S1
58,female,6,MRI,4,2022,5,1.0,...,2.5
```

Optional columns: `filename` (fuses the image embedding),
`baseline_grade_*` (enables progression-risk supervision).

> **Note:** DDD and spondylolisthesis heads stay **untrained** until you add
> their label files — predictions then show a warning instead of fake numbers.
