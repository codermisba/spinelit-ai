"""
train_calibrator.py
===================

Fit real calibration (isotonic regression) mapping raw per-level Pfirrmann
model confidence -> observed probability of DDD (grade >= 2), using labelled
data from the SPIDER dataset.

Run on Colab (GPU) after training `train.py` with DDD (Pfirrmann) labels, OR
whenever you have a validation set with labels. Saves artifacts to
CALIB_ARTIFACT so the pipeline reports *calibrated* probabilities instead of
the deterministic fallback.

Expected label CSV (same schema train.py uses):
    dataset/ddd_labels.csv      : filename,level,pfirrmann_grade

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
)
from calibration import _Calibrators
from utils import decode_outputs, load_model, preprocess_image


def _load_labels(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["filename"] = df["filename"].astype(str).str.strip()
    return df


def _collect(csv_path: Path) -> list:
    """Return list of (image_names, levels, pfirrmann_grade) from the CSV."""
    df = _load_labels(csv_path)
    if df is None:
        return []
    rows = []
    for _, row in df.iterrows():
        name = row["filename"]
        found = None
        for folder in Path(DATA_DIR).glob("*_jpgs"):
            candidate = folder / name
            if candidate.exists():
                found = candidate
                break
        if found is None:
            continue
        rows.append((str(name), str(row["level"]).strip(),
                     int(row["pfirrmann_grade"])))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to model checkpoint (default best_model.pth)")
    args = ap.parse_args()

    model, device, ckpt = load_model(args.checkpoint or "checkpoints/best_model.pth")
    model.eval()
    print(f"Loaded : {ckpt}")

    calibrators = _Calibrators()

    rows = _collect(DDD_LABELS_CSV)
    if rows:
        raw, y = [], []
        for name, level, grade in rows:
            img_path = _find_image(name)
            if img_path is None:
                continue
            img = Image.open(img_path).convert("RGB")
            outputs = model(preprocess_image(img).unsqueeze(0).to(device))
            dec = decode_outputs(outputs)
            idx = DISC_LEVELS.index(level)
            y_disease = 1.0 if grade >= 2 else 0.0   # Pfirrmann 2+ = DDD
            raw.append(float(dec["ddd_conf"][idx]))
            y.append(y_disease)
        calibrators.fit_ddd(raw, y)
        print(f"Fitted DDD calibrator on {len(raw)} samples.")

    state = {
        "ddd": calibrators.ddd,
        "source": "fitted",
    }
    CALIB_ARTIFACT.parent.mkdir(exist_ok=True)
    with open(CALIB_ARTIFACT, "wb") as fh:
        pickle.dump(state, fh)
    print(f"Saved calibrators -> {CALIB_ARTIFACT}")


def _find_image(name):
    for folder in Path(DATA_DIR).glob("*_jpgs"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
