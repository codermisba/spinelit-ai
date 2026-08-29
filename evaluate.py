"""
evaluate.py
===========

Evaluate the Spine Foundation Model on the validation split.

Reports:
- Coordinate MAE / MSE (normalized)
- Mean + per-point localization error (px) — vertebrae and discs separate
- Confidence statistics
- DDD Pfirrmann accuracy and grade MAE (only when labels exist)

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

from config import (
    BATCH_SIZE,
    DISC_LEVELS,
    IMAGE_SIZE,
    NUM_KEYPOINTS,
    NUM_WORKERS,
    PIN_MEMORY,
    VERTEBRAE,
)
from dataset import SpineDataset
from model import SpineFoundationModel
from utils import severity_from_grade

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIXEL_SCALE = IMAGE_SIZE


def build_val_indices(dataset: SpineDataset) -> np.ndarray:
    """Recompute the same stratified validation split used by train.py."""
    has_source = "source" in dataset.disc_csv.columns
    if has_source:
        labels = [
            dataset.groups.get_group(fn).iloc[0]["source"]
            for fn in dataset.image_names
        ]
    else:
        labels = list(range(len(dataset.image_names)))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                      random_state=42)
    _, val_idx = next(splitter.split(dataset.image_names, labels))
    return val_idx


@torch.no_grad()
def compute_metrics(model, val_loader, max_samples=None) -> dict:

    coord_errors, confs = [], []
    ddd_acc, ddd_mae = [], []

    model.eval()

    for batch in val_loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        targets = batch["coords"].to(DEVICE, non_blocking=True)
        visible = batch["point_visible"].to(DEVICE, non_blocking=True)

        outputs = model(images)

        preds = outputs["coords"].float().view(-1, NUM_KEYPOINTS, 2).cpu()
        targs = targets.float().view(-1, NUM_KEYPOINTS, 2).cpu()
        vis = visible.cpu()

        errors = torch.sqrt(((preds - targs) ** 2).sum(-1) + 1e-12)
        errors = torch.where(
            vis.bool(), errors, torch.tensor(float("nan"))
        )
        coord_errors.append(errors.numpy())
        confs.append(outputs["localization_conf"].float().cpu().numpy())

        if batch["ddd_mask"].sum() > 0:
            mask = batch["ddd_mask"].bool()
            ddd_class = batch["ddd_class"].long()
            pred_class = outputs["ddd_logits"].argmax(dim=-1).cpu().long()
            pred_grade = (pred_class + 1).float()
            tgt_grade = (ddd_class + 1).float()
            ddd_acc.append((pred_class == ddd_class)[mask].float().numpy())
            ddd_mae.append((pred_grade - tgt_grade).abs()[mask].numpy())

        if max_samples and len(coord_errors) * BATCH_SIZE >= max_samples:
            break

    point_errors = np.concatenate(coord_errors)      # (N, 10), NaN = masked
    confidences = np.concatenate(confs)

    n_v = len(VERTEBRAE)

    results = {
        "num_points": int(np.sum(~np.isnan(point_errors))),
        "mean_loc_err_px": float(np.nanmean(point_errors) * PIXEL_SCALE),
        "vertebra_err_px": float(
            np.nanmean(point_errors[:, :n_v]) * PIXEL_SCALE
        ),
        "disc_err_px": float(
            np.nanmean(point_errors[:, n_v:]) * PIXEL_SCALE
        ),
        "per_point_px": np.nanmean(point_errors, axis=0) * PIXEL_SCALE,
        "conf_mean": float(confidences.mean()),
        "conf_per_point": confidences.mean(axis=0),
    }

    results["ddd_accuracy"] = (
        float(np.concatenate(ddd_acc).mean())
        if ddd_acc else None
    )
    results["ddd_mae_grade"] = (
        float(np.concatenate(ddd_mae).mean())
        if ddd_mae else None
    )

    return results


def print_summary(results: dict) -> None:

    print()
    print("## Foundation Model Evaluation")
    print()
    print(f"Visible Points Evaluated : {results['num_points']}")
    print()
    print(f"Mean Localization Error (px) : {results['mean_loc_err_px']:.2f}")
    print(f"Vertebrae Mean Error (px)    : {results['vertebra_err_px']:.2f}")
    print(f"Disc Mean Error (px)         : {results['disc_err_px']:.2f}")
    print()

    print("Per-Point Localization Error (px):")
    print()
    for i, name in enumerate(list(VERTEBRAE) + list(DISC_LEVELS)):
        kind = "vertebra" if i < len(VERTEBRAE) else "disc"
        print(f"{name:<6} ({kind:<8}) : {results['per_point_px'][i]:6.2f}"
              f"   conf {results['conf_per_point'][i]:.3f}")

    print()
    print(f"Mean Confidence : {results['conf_mean']:.3f}")

    if results["ddd_accuracy"] is not None:
        print(f"DDD Pfirrmann Accuracy : "
              f"{results['ddd_accuracy'] * 100:.2f} %")
        print(f"DDD Grade MAE          : "
              f"{results['ddd_mae_grade']:.3f} Pfirrmann grades")

    print()


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(
            f"No trained model checkpoint at {checkpoint_path}."
            f" Train first: python train.py"
        )
        return

    print(f"Using Device : {DEVICE}")
    print(f"Checkpoint   : {checkpoint_path}")

    dataset = SpineDataset()
    val_dataset = Subset(dataset, build_val_indices(dataset))

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = SpineFoundationModel().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = compute_metrics(model, val_loader, args.max_samples)

    print_summary(results)


if __name__ == "__main__":
    main()
