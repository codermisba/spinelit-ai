"""
train_calibrator.py
===================

Fit real calibration (isotonic regression) for the DDD head and measure it on
held-out cases.

What is calibrated
------------------
The model outputs a 5-class Pfirrmann distribution per disc.  The quantity
that matters clinically is **P(degeneration) = P(Pfirrmann >= 2)**, i.e. the
probability mass on grades 2-5.  The isotonic fit maps that raw probability
to the observed frequency of DDD (grade >= 2) in SPIDER.

Honest evaluation
-----------------
The calibrator is fitted on 80% of *cases* and scored on the held-out 20%
(Brier score, expected calibration error).  Fitting and scoring on the same
rows would flatter the numbers, so the report separates them.

Run after the DDD head is trained (or on Colab/GPU):

    python train_calibrator.py
    python train_calibrator.py --checkpoint checkpoints/best_model.pth

Writes ``CALIB_ARTIFACT``; ``calibration.py`` loads it automatically.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from utils import decode_outputs, load_model, predict, preprocess_image


def _find_image(name: str):
    for folder in Path(DATA_DIR).glob("*_jpgs"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _collect(csv_path: Path) -> dict[str, list[tuple[str, int]]]:
    """Group labels by image: {filename: [(level, grade), ...]}."""
    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].astype(str).str.strip()
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["filename"]].append(
            (str(row["level"]).strip(), int(row["pfirrmann_grade"]))
        )
    return dict(grouped)


def _brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not sel.any():
            continue
        total += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="path to model checkpoint (default best_model.pth)")
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="Fraction of cases held out to measure calibration.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, device, ckpt, meta = load_model(
        args.checkpoint or "checkpoints/best_model.pth")
    model.eval()
    print(f"Loaded : {ckpt}")

    tasks = dict(meta.get("tasks") or {})
    if not tasks.get("ddd", False):
        print(
            "[skip] This checkpoint has no trained DDD head "
            "(dataset/ddd_labels.csv was absent at training time). "
            "Nothing to calibrate; keeping the existing calibrator."
        )
        return

    if not Path(DDD_LABELS_CSV).exists():
        print(f"[skip] Missing {DDD_LABELS_CSV}.")
        return

    grouped = _collect(DDD_LABELS_CSV)
    names = sorted(grouped)
    if len(names) < 10:
        print(f"[skip] Only {len(names)} labelled cases; need >= 10.")
        return

    rng = np.random.default_rng(args.seed)
    order = list(names)
    rng.shuffle(order)
    n_val = max(1, int(round(args.val_fraction * len(order))))
    val_names = set(order[:n_val])
    fit_names = set(order[n_val:])

    # Predict once per image (209 forwards), not once per level (1045).
    raw, y, split = [], [], []
    for i, name in enumerate(names):
        path = _find_image(name)
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        outputs = predict(model, preprocess_image(img), device)
        dec = decode_outputs(outputs)
        row_split = "val" if name in val_names else "fit"
        for level, grade in grouped[name]:
            if level not in DISC_LEVELS:
                continue
            idx = DISC_LEVELS.index(level)
            probs = np.asarray(dec["ddd_prob"][idx], dtype=float)
            raw.append(float(np.clip(1.0 - probs[0], 0.0, 1.0)))  # P(grade>=2)
            y.append(1.0 if grade >= 2 else 0.0)
            split.append(row_split)
        if (i + 1) % 40 == 0:
            print(f"  predicted {i + 1}/{len(names)} cases")

    raw = np.asarray(raw)
    y = np.asarray(y)
    is_val = np.asarray([s == "val" for s in split])

    if len(raw) < 50 or is_val.sum() < 20:
        print(f"[skip] Too few samples ({len(raw)}, val {int(is_val.sum())}).")
        return

    calibrators = _Calibrators()
    calibrators.fit_ddd(raw[~is_val], y[~is_val])

    # Uncalibrated vs calibrated on the held-out cases.
    p_raw_val = raw[is_val]
    p_cal_val = np.clip(calibrators.ddd.predict(raw[is_val]), 0.0, 1.0)
    y_val = y[is_val]
    base = float(y_val.mean())
    print()
    print(f"Samples : {len(raw)} discs over {len(names)} cases "
          f"(fit {int((~is_val).sum())}, val {int(is_val.sum())})")
    print(f"DDD prevalence (val)      : {base:.3f}")
    print(f"Brier  raw / calibrated   : {_brier(y_val, p_raw_val):.4f} / "
          f"{_brier(y_val, p_cal_val):.4f}   (lower is better)")
    print(f"ECE    raw / calibrated   : {_ece(y_val, p_raw_val):.4f} / "
          f"{_ece(y_val, p_cal_val):.4f}   (lower is better)")

    # A correction that worsens held-out Brier is not shipped.  Raw
    # P(Pfirrmann >= 2) comes straight from the softmax, so keeping it is both
    # honest and simpler than applying a fitted map that overfits the fit set.
    keep_raw = _brier(y_val, p_cal_val) >= _brier(y_val, p_raw_val)
    if keep_raw:
        print("Using raw P(DDD) probabilities: the isotonic fit did not "
              "improve held-out Brier.")
    else:
        print("Using fitted isotonic calibrator (improved held-out Brier).")
    print("Reliability (held-out, calibrated):")
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        sel = (p_cal_val >= lo) & (p_cal_val < hi)
        if sel.any():
            print(f"  P in [{lo:.1f},{min(hi,1.0):.1f})  n={int(sel.sum()):4d}  "
                  f"predicted {p_cal_val[sel].mean():.3f}  "
                  f"observed {y_val[sel].mean():.3f}")

    state = {
        "ddd": None if keep_raw else calibrators.ddd,
        "identity": bool(keep_raw),
        "source": "raw-passthrough" if keep_raw else "fitted-heldout",
        "target": "P(Pfirrmann >= 2)",
        "n_fit": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
        "val_prevalence": base,
        "brier_raw": _brier(y_val, p_raw_val),
        "brier_calibrated": _brier(y_val, p_cal_val),
        "ece_raw": _ece(y_val, p_raw_val),
        "ece_calibrated": _ece(y_val, p_cal_val),
        "checkpoint": str(ckpt),
    }
    CALIB_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_ARTIFACT, "wb") as fh:
        pickle.dump(state, fh)
    print(f"\nSaved calibration artifact -> {CALIB_ARTIFACT}")


if __name__ == "__main__":
    main()
