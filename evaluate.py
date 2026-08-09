"""
evaluate.py
===========

Evaluate the Spine Foundation Model on the validation split.

Loads checkpoints/best_model.pth (or a checkpoint passed with --checkpoint),
runs inference on the same validation split used during training and reports:

- Coordinate MAE / MSE (normalized 0-1 units)
- Mean Euclidean localization error (pixels, at 512px scale)
- Per-level localization error
- Confidence statistics

Usage
-----
python evaluate.py
python evaluate.py --checkpoint checkpoints/last_model.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

from config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, PIN_MEMORY
from dataset import SpineDataset
from model import SpineFoundationModel
from utils import LEVELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIXEL_SCALE = IMAGE_SIZE  # training resolution used to convert normalized -> pixels


# ---------------------------------------------------------
# Validation split (must match train.py exactly)
# ---------------------------------------------------------

def build_val_indices(dataset: SpineDataset) -> np.ndarray:
    """Recompute the same stratified validation split used by train.py."""
    labels = []

    for filename in dataset.image_names:
        rows = dataset.groups.get_group(filename)
        labels.append(rows.iloc[0]["source"])

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    _, val_idx = next(splitter.split(dataset.image_names, labels))

    return val_idx


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_metrics(
    model: SpineFoundationModel,
    val_loader: DataLoader,
    max_samples: int | None = None,
) -> dict:
    """Run inference on the validation loader and compute all metrics."""

    coord_errors = []      # euclidean error per point, normalized
    mae_squared = []       # per-axis squared errors (MSE)
    mae_abs = []           # per-axis absolute errors (MAE)
    confidences = []       # confidence per level per sample

    model.eval()

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE, non_blocking=True)
            targets = batch["coords"].to(DEVICE, non_blocking=True)

            outputs = model(images)

            preds = outputs["coords"].cpu().numpy()
            confs = outputs["confidence"].cpu().numpy()
            targs = targets.cpu().numpy()

            num_points = preds.shape[1] // 2

            pred_points = preds.reshape(-1, num_points, 2)
            target_points = targs.reshape(-1, num_points, 2)

            diff = pred_points - target_points

            # euclidean error per point (normalized 0-1)
            coord_errors.append(
                np.sqrt((diff ** 2).sum(axis=-1))
            )

            # per-axis errors for MAE / MSE
            flat = diff.reshape(-1, 2)
            mae_squared.append((flat ** 2))
            mae_abs.append(np.abs(flat))

            confidences.append(confs)

            if max_samples is not None:
                processed = (
                    np.concatenate(coord_errors).shape[0]
                )
                if processed >= max_samples:
                    break

    coord_errors = np.concatenate(coord_errors)          # (N, 5)
    mse = np.mean(np.concatenate(mae_squared))           # normalized units
    mae = np.mean(np.concatenate(mae_abs))               # normalized units
    confidences = np.concatenate(confidences)            # (N, 5)

    results = {
        "num_samples": coord_errors.shape[0],
        "coord_mae": mae,
        "coord_mse": mse,
        "mean_loc_err_norm": float(coord_errors.mean()),
        "mean_loc_err_px": float(coord_errors.mean() * PIXEL_SCALE),
        "per_level_px": coord_errors.mean(axis=0) * PIXEL_SCALE,
        "per_level_norm": coord_errors.mean(axis=0),
        "conf_mean": float(confidences.mean()),
        "conf_per_level": confidences.mean(axis=0),
        "conf_std": float(confidences.std()),
    }

    return results


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def print_summary(results: dict) -> None:

    print()
    print("## Foundation Model Evaluation")
    print()
    print(f"Validation Samples : {results['num_samples']}")
    print()
    print(f"Coordinate MAE (0-1) : {results['coord_mae']:.4f}")
    print(f"Coordinate MSE (0-1) : {results['coord_mse']:.6f}")
    print()
    print(
        f"Mean Localization Error (px) : "
        f"{results['mean_loc_err_px']:.2f}"
    )
    print(
        f"Mean Localization Error (0-1): "
        f"{results['mean_loc_err_norm']:.4f}"
    )
    print()

    print("Per-Level Localization Error (px):")
    print()

    for i, level in enumerate(LEVELS):
        print(
            f"{level:<6} : "
            f"{results['per_level_px'][i]:.2f}"
        )

    print()

    print("Confidence Statistics:")
    print()

    for i, level in enumerate(LEVELS):
        print(
            f"{level:<6} : "
            f"{results['conf_per_level'][i]:.3f}"
        )

    print()
    print(f"Mean Confidence        : {results['conf_mean']:.3f}")
    print(f"Confidence Std         : {results['conf_std']:.3f}")
    print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Evaluate the Spine Foundation Model."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the checkpoint to evaluate.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only evaluate this many samples (quick smoke test).",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(
            f"No trained model checkpoint found at {checkpoint_path}. "
            f"Please train the model first: python train.py"
        )
        return

    print(f"Using Device  : {DEVICE}")
    print(f"Checkpoint    : {checkpoint_path}")

    print("\nLoading dataset...")
    dataset = SpineDataset()

    val_idx = build_val_indices(dataset)
    val_dataset = Subset(dataset, val_idx)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Validation Images : {len(val_dataset)}")

    print("\nLoading model...")
    model = SpineFoundationModel().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    results = compute_metrics(model, val_loader, args.max_samples)

    print_summary(results)


if __name__ == "__main__":

    main()
