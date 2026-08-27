"""
train_calibrator.py
===================

Fit real calibration (isotonic regression) mapping raw per-level model
confidence -> observed probability of disease, using LABELLED data.

Run on Colab (GPU) after training `train.py` with DDD and
spondylolisthesis labels, OR whenever you have a validation set with
labels. Saves artifacts to CALIB_ARTIFACT so the pipeline reports
*calibrated* probabilities instead of the deterministic fallback.

Expected label CSVs (same schema train.py uses):
    dataset/ddd_labels.csv      : filename,level,grade
    dataset/spondy_labels.csv   : filename,level,slip_percent

Usage
-----
python train_calibrator.py [--checkpoint path] [--split val]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from config import (
    CALIB_ARTIFACT,
    DATA_DIR,
    DDD_LABELS_CSV,
    DISC_LEVELS,
    SPONDY_LABELS_CSV,
)
from calibration import _Calibrators
from utils import decode_outputs, load_model, preprocess_image


def _load_labels(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["filename"] = df["filename"].astype(str).str.strip()
    return df


def _collect(csv_path: Path, value_col):
    """Return (image_names, levels, values) from a label CSV."""
    df = _load_labels(csv_path)
    if df is None:
        return [], [], []
    names, levels, values = [], [], []
    for _, row in df.iterrows():
        name = row["filename"]
        # resolve image location across known image dirs
        found = None
        for folder in Path(DATA_DIR).glob("*_jpgs"):
            candidate = folder / name
            if candidate.exists():
                found = candidate
                break
        if found is None:
            continue
        names.append(name)
        levels.append(str(row["level"]).strip())
        values.append(float(row[value_col]))
    return names, levels, values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to model checkpoint (default best_model.pth)")
    args = ap.parse_args()

    model, device, ckpt = load_model(args.checkpoint or "checkpoints/best_model.pth")
    model.eval()
    print(f"Loaded : {ckpt}")

    calibrators = _Calibrators()

    ddd_df = _load_labels(DDD_LABELS_CSV)
    sp_df = _load_labels(SPONDY_LABELS_CSV)

    if ddd_df is not None:
        rows = _collect(DDD_LABELS_CSV, "grade")
        if rows[0]:
            raw, y = [], []
            for name, level, grade in zip(*rows):
                img_path = _find_image(name)
                if img_path is None:
                    continue
                img = Image.open(img_path).convert("RGB")
                outputs = model(preprocess_image(img).unsqueeze(0).to(device))
                dec = decode_outputs(outputs)
                idx = DISC_LEVELS.index(level)
                y_disease = 1.0 if grade > 0 else 0.0
                raw.append(float(dec["ddd_conf"][idx]))
                y.append(y_disease)
            calibrators.fit_ddd(raw, y)
            print(f"Fitted DDD calibrator on {len(raw)} samples.")

    if sp_df is not None:
        rows = _collect(SPONDY_LABELS_CSV, "slip_percent")
        if rows[0]:
            raw, y = [], []
            for name, level, slip in zip(*rows):
                img_path = _find_image(name)
                if img_path is None:
                    continue
                img = Image.open(img_path).convert("RGB")
                outputs = model(preprocess_image(img).unsqueeze(0).to(device))
                dec = decode_outputs(outputs)
                idx = DISC_LEVELS.index(level)
                y_disease = 1.0 if slip > 0 else 0.0
                raw.append(float(dec["spondy_conf"][idx]))
                y.append(y_disease)
            calibrators.fit_spondy(raw, y)
            print(f"Fitted spondy calibrator on {len(raw)} samples.")

    state = {
        "ddd": calibrators.ddd,
        "spondy": calibrators.spondy,
        "source": "fitted",
    }
    CALIB_ARTIFACT.parent.mkdir(exist_ok=True)
    with open(CALIB_ARTIFACT, "wb") as fh:
        pickle.dump(state, fh)
    print(f"Saved calibrators -> {CALIB_ARTIFACT}")


def _find_image(name):
    # NOTE: helper treated as local to keep the module portable.
    for folder in Path(DATA_DIR).glob("*_jpgs"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
