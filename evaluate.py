"""
evaluate.py
===========

Evaluate the Spine Foundation Model on the validation split.

Reports:
- Coordinate MAE / MSE (normalized)
- Mean + per-point localization error (px) — vertebrae and discs separate
- PCK@5% / PCK@10% (percentage of landmarks within 5% / 10% of image size)
- Confidence statistics + mean heatmap peak
- DDD Pfirrmann accuracy and grade MAE — only when the checkpoint actually
  contains a trained DDD head and the dataset has labels

The checkpoint is loaded through ``utils.load_model`` so the training-time
image size, backbone name and normalisation are applied automatically.

Usage
-----
python evaluate.py
python evaluate.py --checkpoint checkpoints/last_model.pth
python evaluate.py --max-samples 200
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

import config as _cfg
from config import (
    BATCH_SIZE,
    DISC_LEVELS,
    NUM_KEYPOINTS,
    NUM_WORKERS,
    PIN_MEMORY,
    VERTEBRAE,
)
from dataset import SpineDataset
from utils import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_labels(dataset: SpineDataset) -> list:
    """Per-file stratification label, matching train.py's rule."""
    import collections
    if "source" in dataset.disc_csv.columns:
        labels = [
            dataset.groups.get_group(fn).iloc[0]["source"]
            for fn in dataset.image_names
        ]
        return _shrink_rare_labels(labels)
    ddd = dataset.ddd_labels
    if ddd is not None and len(ddd):
        labels = []
        for fn in dataset.image_names:
            rows = ddd[ddd["filename"].astype(str) == str(fn)]
            if len(rows) == 0:
                labels.append("nograde")
            else:
                labels.append(str(int(round(rows["pfirrmann_grade"]
                                           .astype(float).median()))))
        return _shrink_rare_labels(labels)
    raise ValueError("No 'source' column and no DDD labels for split stratify.")


def _shrink_rare_labels(labels):
    from collections import Counter
    counts = Counter(labels)
    rare = {lab for lab, n in counts.items() if n < 2}
    return [("other" if lab in rare else lab) for lab in labels]


def build_val_indices(dataset: SpineDataset) -> np.ndarray:
    """Recompute the *exact* validation split used by train.py.

    The split must be identical, otherwise evaluate.py would report metrics
    on images the model trained on.  Keep this in lock-step with
    Train.prepare_dataloaders (15% validation, same seed, same stratified
    labels, same fallback).
    """
    labels = _split_labels(dataset)
    n_images = len(dataset.image_names)
    n_classes = len(set(labels))
    n_val = max(1, int(round(0.15 * n_images)))
    if n_images >= 20 and n_val >= n_classes:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                          random_state=42)
        _, val_idx = next(splitter.split(dataset.image_names, labels))
        return np.asarray(val_idx)
    perm = torch.randperm(
        n_images, generator=torch.Generator().manual_seed(42)
    ).tolist()
    return np.asarray(perm[:n_val])


@torch.no_grad()
def compute_metrics(model, val_loader, max_samples=None,
                    pixel_scale: int | None = None,
                    ddd_available: bool = False) -> dict:

    pixel_scale = int(pixel_scale or _cfg.IMAGE_SIZE)
    coord_errors, confs, peaks = [], [], []
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
        if "heatmap_peak" in outputs:
            peaks.append(outputs["heatmap_peak"].float().cpu().numpy())

        if ddd_available and batch["ddd_mask"].sum() > 0:
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
    valid = ~np.isnan(point_errors)
    finite = point_errors[valid]

    n_v = len(VERTEBRAE)

    results = {
        "num_points": int(valid.sum()),
        "image_size": pixel_scale,
        "mean_loc_err_px": float(np.nanmean(point_errors) * pixel_scale),
        "vertebra_err_px": float(
            np.nanmean(point_errors[:, :n_v]) * pixel_scale
        ),
        "disc_err_px": float(
            np.nanmean(point_errors[:, n_v:]) * pixel_scale
        ),
        "per_point_px": np.nanmean(point_errors, axis=0) * pixel_scale,
        "mean_loc_err_norm": float(np.nanmean(point_errors)),
        "pck5": float((finite <= 0.05).mean()) if finite.size else None,
        "pck10": float((finite <= 0.10).mean()) if finite.size else None,
        "conf_mean": float(confidences.mean()),
        "conf_per_point": confidences.mean(axis=0),
    }

    if peaks:
        results["heatmap_peak_mean"] = float(np.concatenate(peaks).mean())

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
    print(f"Image size (checkpoint) : {results['image_size']} px")
    print(f"Visible Points Evaluated : {results['num_points']}")
    print()
    print(f"Mean Localization Error (px) : {results['mean_loc_err_px']:.2f}")
    print(f"Vertebrae Mean Error (px)    : {results['vertebra_err_px']:.2f}")
    print(f"Disc Mean Error (px)         : {results['disc_err_px']:.2f}")
    print(f"Mean normalized error        : {results['mean_loc_err_norm']:.4f}")
    if results["pck5"] is not None:
        print(f"PCK@5%  (<= 5% image size)   : {results['pck5'] * 100:.2f} %")
        print(f"PCK@10% (<= 10% image size)  : {results['pck10'] * 100:.2f} %")
    print()

    print("Per-Point Localization Error (px):")
    print()
    for i, name in enumerate(list(VERTEBRAE) + list(DISC_LEVELS)):
        kind = "vertebra" if i < len(VERTEBRAE) else "disc"
        print(f"{name:<6} ({kind:<8}) : {results['per_point_px'][i]:6.2f}"
              f"   conf {results['conf_per_point'][i]:.3f}")

    print()
    print(f"Mean Confidence : {results['conf_mean']:.3f}")
    if "heatmap_peak_mean" in results:
        print(f"Mean heatmap peak : {results['heatmap_peak_mean']:.3f}")

    if results["ddd_accuracy"] is not None:
        print(f"DDD Pfirrmann Accuracy : "
              f"{results['ddd_accuracy'] * 100:.2f} %")
        print(f"DDD Grade MAE          : "
              f"{results['ddd_mae_grade']:.3f} Pfirrmann grades")
    else:
        print("DDD Pfirrmann metrics  : unavailable "
              "(no trained DDD head and/or no labels)")

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

    # load_model applies the checkpoint's own image size / backbone / norm.
    model, device, _, meta = load_model(str(checkpoint_path))
    tasks = dict(meta.get("tasks") or {})
    image_size = int(meta.get("image_size") or _cfg.IMAGE_SIZE)

    dataset = SpineDataset()
    val_dataset = Subset(dataset, build_val_indices(dataset))

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    results = compute_metrics(
        model, val_loader, args.max_samples,
        pixel_scale=image_size,
        ddd_available=bool(tasks.get("ddd", False)),
    )

    print_summary(results)


if __name__ == "__main__":
    main()
