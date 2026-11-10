"""
finetune_ddd.py
===============

Train the Pfirrmann (DDD) head without touching — or even re-running — the
localizer backbone.

Why a dedicated fine-tune
-------------------------
``train.py`` can jointly train localization + DDD, but re-running it from
scratch would throw away the validated 256 px localizer (2.35 px mean error /
97.9% PCK@10).  The DDD head only needs the frozen FPN feature map plus the
(already accurate) disc centres, so we:

1. freeze the backbone + FPN,
2. cache their outputs **once** for every labelled SPIDER case,
3. train only ``disc_head`` on the cached features (seconds per epoch),
4. merge the trained head back and set ``tasks.ddd = true``.

Caching matters: the ConvNeXt forward pass dominates CPU time, so the previous
uncached run only completed ~20 optimizer steps per minute and never converged
(train accuracy stayed at chance).  With cached features the same data is
reused for hundreds of epochs.

The head reads three complementary signals:

* Gaussian-pooled FPN features around each disc centre (context/shape),
* a raw-image patch encoder + brightness statistics (the actual T2 nucleus
  pulposus signal Pfirrmann grading is based on),
* relative disc spacing (disc-height loss is part of the grading scale).

``disc_pool_sigma_fraction`` and the patch geometry are written into the
checkpoint metadata so inference reproduces the training configuration.

Usage
-----
    python finetune_ddd.py                     # train + merge into best_model.pth
    python finetune_ddd.py --epochs 120
    python finetune_ddd.py --output checkpoints/ddd_experiment.pth --epochs 40

Output
------
``checkpoints/best_model.pth`` (and ``last_model.pth``) with a trained DDD
head and ``tasks.ddd = true``.  The original localizer checkpoint is backed up
to ``checkpoints/best_model_localizer.pth`` the first time this runs.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset

import config as _cfg
from config import (BEST_MODEL, DDD_LABELS_CSV, LAST_MODEL,
                    NUM_PFRRMANN_CLASSES)
from dataset import SpineDataset
from model import (DISC_PATCH_DIM, DISC_PATCH_HALF_H, DISC_PATCH_HALF_W,
                   DISC_PATCH_RES, DISC_PATCH_STATS, _soft_argmax_2d)
from utils import load_model

HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 5.0
VAL_FRACTION = 0.20
SEED = 42
PATIENCE = 25


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def merge_tasks(checkpoint: dict, ddd: bool = True) -> dict:
    """Set ``ddd`` on a checkpoint's task flags without dropping others.

    The localizer checkpoint also carries ``localization: true`` in both the
    top-level ``tasks`` block and ``meta.tasks``.  Overwriting the dict with
    ``{"ddd": True}`` made the API report localization as unavailable even
    though the same weights just localized every landmark.
    """
    meta_tasks = dict((checkpoint.get("meta") or {}).get("tasks") or {})
    top_tasks = dict(checkpoint.get("tasks") or {})
    tasks = {**meta_tasks, **top_tasks}
    tasks["ddd"] = bool(ddd)
    tasks.setdefault("localization", True)
    tasks["seg"] = bool(tasks.get("seg", False))
    return tasks


def _spider_indices(dataset: SpineDataset) -> list[int]:
    """Indices of cases that actually have Pfirrmann labels."""
    if dataset.ddd_labels is None:
        return []
    labelled = set(dataset.ddd_labels["filename"].astype(str))
    return [i for i, name in enumerate(dataset.image_names)
            if str(name) in labelled]


def _case_labels(dataset: SpineDataset, indices: list[int]) -> list[str]:
    labels = []
    for i in indices:
        name = dataset.image_names[i]
        rows = dataset.ddd_labels[dataset.ddd_labels["filename"].astype(str)
                                   == str(name)]
        grade = int(round(float(rows["pfirrmann_grade"].median())))
        labels.append(str(grade))
    return labels


def _class_weights(dataset: SpineDataset, indices: list[int]) -> torch.Tensor:
    counts = Counter()
    for i in indices:
        name = dataset.image_names[i]
        rows = dataset.ddd_labels[dataset.ddd_labels["filename"].astype(str)
                                   == str(name)]
        for value in rows["pfirrmann_grade"].astype(int):
            counts[int(value)] += 1
    total = sum(counts.values())
    weights = []
    for grade in range(1, NUM_PFRRMANN_CLASSES + 1):
        c = counts.get(grade, 0)
        # Mild inverse-frequency weighting: keep rare grades learnable without
        # collapsing the majority classes.
        weights.append((total / (NUM_PFRRMANN_CLASSES * c)) ** 0.5 if c else 0.0)
    w = torch.tensor(weights, dtype=torch.float32)
    present = torch.tensor([counts.get(g, 0) > 0
                            for g in range(1, NUM_PFRRMANN_CLASSES + 1)])
    return w / w[present].mean()


# ----------------------------------------------------------------------
# Feature cache
# ----------------------------------------------------------------------

@torch.no_grad()
def build_cache(model, dataset: SpineDataset, indices: list[int],
                device, batch_size: int = 8) -> dict:
    """Run the frozen backbone once and cache everything the head needs."""
    model.eval()
    images, fmaps, coords, classes, masks = [], [], [], [], []
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False)
    for batch in loader:
        x = batch["image"].to(device)
        fmap = model.fpn(model.backbone(x))
        images.append(x.cpu())
        fmaps.append(fmap.cpu())
        coords.append(batch["coords"].float().cpu())
        classes.append(batch["ddd_class"].long().cpu())
        masks.append(batch["ddd_mask"].float().cpu())
    return {
        "images": torch.cat(images),
        "fmaps": torch.cat(fmaps),
        "coords": torch.cat(coords).view(-1, 10, 2),
        "ddd_class": torch.cat(classes),
        "ddd_mask": torch.cat(masks),
        "position": {idx: pos for pos, idx in enumerate(indices)},
    }


def _batch_from_cache(cache: dict, positions: list[int], device,
                      generator: torch.Generator | None = None,
                      augment: bool = False) -> dict:
    pos = torch.tensor(positions, dtype=torch.long)
    x = cache["images"][pos].to(device)
    fmap = cache["fmaps"][pos].to(device)
    coords = cache["coords"][pos].to(device)
    ddd_class = cache["ddd_class"][pos].to(device)
    ddd_mask = cache["ddd_mask"][pos].to(device)

    if augment:
        flip = torch.rand(len(positions), generator=generator) < 0.5
        if flip.any():
            x[flip] = torch.flip(x[flip], dims=[3])
            fmap[flip] = torch.flip(fmap[flip], dims=[3])
            coords = coords.clone()
            coords[flip, :, 0] = 1.0 - coords[flip, :, 0]
    return {
        "image": x, "fmap": fmap, "coords": coords,
        "ddd_class": ddd_class, "ddd_mask": ddd_mask,
    }


def _disc_centers(coords: torch.Tensor) -> torch.Tensor:
    # Dataset order is 5 vertebral centres then 5 disc centres.
    return coords[:, 5:].reshape(-1, 5, 2)


# ----------------------------------------------------------------------
# Train / evaluate on cached features
# ----------------------------------------------------------------------

def _loss_and_acc(model, batch, class_weights, label_smoothing):
    logits = model.disc_head(
        batch["fmap"], _disc_centers(batch["coords"]), image=batch["image"]
    )
    mask = batch["ddd_mask"].bool()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = batch["ddd_class"].reshape(-1)
    flat_mask = mask.reshape(-1)
    loss = F.cross_entropy(
        flat_logits[flat_mask], flat_target[flat_mask],
        weight=class_weights, label_smoothing=label_smoothing,
    )
    pred = logits.argmax(dim=-1)
    diff = (pred[mask] - batch["ddd_class"][mask]).abs()
    correct = int((diff == 0).sum().item())
    within_one = int((diff <= 1).sum().item())
    abs_err = float(diff.sum().item())
    n = int(mask.sum().item())
    return loss, correct, within_one, abs_err, n


@torch.no_grad()
def evaluate(model, cache: dict, positions: list[int], device,
             class_weights, teacher_forced: bool = True) -> dict:
    model.eval()
    model.disc_head.eval()
    loader = DataLoader(positions, batch_size=8, shuffle=False)
    total_loss = correct = total = 0
    abs_err = 0.0
    within_one = 0

    for chunk in loader:
        batch = _batch_from_cache(cache, list(chunk), device)
        if not teacher_forced:
            # Realistic inference: localizer-predicted disc centres.
            logits = model.keypoint_head(batch["fmap"])
            coords, _ = _soft_argmax_2d(logits, model.softargmax_temperature)
            batch["coords"] = coords.view(-1, 10, 2)
        loss, c, w1, e, n = _loss_and_acc(
            model, batch, class_weights, LABEL_SMOOTHING)
        total_loss += float(loss.item())
        correct += c
        within_one += w1
        abs_err += e
        total += n

    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": correct / max(1, total),
        "within_one": within_one / max(1, total),
        "mae": abs_err / max(1, total),
        "n": total,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(BEST_MODEL))
    parser.add_argument("--output", default=str(BEST_MODEL),
                        help="Where to write the merged checkpoint.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=HEAD_LR)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--select-by", choices=("loss", "within_one"),
                        default="loss",
                        help="Validation criterion for keeping the best head. "
                             "'loss' (cross-entropy) is far less noisy than "
                             "exact accuracy on 210 discs.")
    parser.add_argument("--cache-batch-size", type=int, default=8,
                        help="Batch size used once for the backbone cache.")
    args = parser.parse_args()

    if not Path(DDD_LABELS_CSV).exists():
        raise SystemExit(f"Missing {DDD_LABELS_CSV}. Run build_ddd_labels.py.")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading checkpoint : {args.checkpoint}")
    model, device, ckpt_path, meta = load_model(args.checkpoint)
    image_size = int(meta.get("image_size", 256))
    sigma_fraction = float(model.disc_head.sigma_fraction)
    print(f"Device             : {device}")
    print(f"Image size         : {image_size} px")
    print(f"Disc pooling sigma : {sigma_fraction:.3f} of feature map")
    print(f"Existing tasks     : {meta.get('tasks')}")

    # Back up the original localizer checkpoint exactly once.
    backup = Path(ckpt_path).with_name("best_model_localizer.pth")
    if not backup.exists():
        import shutil
        shutil.copy2(ckpt_path, backup)
        print(f"Backed up localizer: {backup}")

    dataset = SpineDataset()
    indices = _spider_indices(dataset)
    if len(indices) < 20:
        raise SystemExit(f"Only {len(indices)} labelled cases found.")
    labels = _case_labels(dataset, indices)
    print(f"Labelled SPIDER cases: {len(indices)} ({Counter(labels)})")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val_fraction, random_state=SEED)
    train_pos, val_pos = next(splitter.split(indices, labels))
    train_idx = [indices[p] for p in train_pos]
    val_idx = [indices[p] for p in val_pos]
    print(f"Train / Val cases    : {len(train_idx)} / {len(val_idx)}")

    print("Caching frozen backbone features (one pass)...")
    cache = build_cache(model, dataset, indices, device,
                        batch_size=args.cache_batch_size)
    cache["position"] = {idx: i for i, idx in enumerate(indices)}
    train_cache_pos = [cache["position"][i] for i in train_idx]
    val_cache_pos = [cache["position"][i] for i in val_idx]
    print(f"Cache tensors      : {tuple(cache['images'].shape)}, "
          f"{tuple(cache['fmaps'].shape)}")

    class_weights = _class_weights(dataset, train_idx).to(device)
    print("Class weights       : "
          + ", ".join(f"{g}:{w:.2f}" for g, w in
                      zip(range(1, 6), class_weights.tolist())))

    # Only the head is optimised; backbone/FPN stay frozen (and are not
    # re-run during training because their outputs are cached).
    for param in model.parameters():
        param.requires_grad = False
    for param in model.disc_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.disc_head.parameters(), lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.1,
    )
    generator = torch.Generator().manual_seed(SEED)

    baseline = evaluate(model, cache, val_cache_pos, device, class_weights)
    print(f"Baseline val DDD acc: {baseline['accuracy']*100:.1f}% "
          f"(random head)")

    best_state = copy.deepcopy(model.disc_head.state_dict())
    # Selection metric: validation cross-entropy by default (stable on only
    # 210 validation discs); within-one-plus-accuracy is available for
    # comparison but fluctuates by several points epoch to epoch.
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_metrics = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.disc_head.train()
        perm = torch.randperm(len(train_cache_pos), generator=generator).tolist()
        running = 0.0
        n_batches = 0
        for start in range(0, len(perm), args.batch_size):
            chunk = [train_cache_pos[p] for p in perm[start:start + args.batch_size]]
            if len(chunk) < 2:
                continue
            batch = _batch_from_cache(cache, chunk, device,
                                      generator=generator, augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _, _ = _loss_and_acc(
                model, batch, class_weights, LABEL_SMOOTHING)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.disc_head.parameters()
                 if p.requires_grad], max_norm=GRAD_CLIP)
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        scheduler.step()
        metrics = evaluate(model, cache, val_cache_pos, device, class_weights)
        if args.select_by == "loss":
            score = (-metrics["loss"], metrics["within_one"],
                     metrics["accuracy"])
        else:
            score = (metrics["within_one"], metrics["accuracy"],
                     -metrics["mae"])
        improved = score > best_score
        if improved:
            best_score = score
            best_metrics = dict(metrics)
            best_epoch = epoch
            best_state = copy.deepcopy(model.disc_head.state_dict())
            stale = 0
        else:
            stale += 1

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train {running/max(1,n_batches):.4f} | "
              f"val {metrics['loss']:.4f} | "
              f"acc {metrics['accuracy']*100:.1f}% | "
              f"+/-1 {metrics['within_one']*100:.1f}% | "
              f"MAE {metrics['mae']:.3f}"
              + ("  *" if improved else ""))
        if stale >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs.")
            break

    if best_metrics is None:
        raise SystemExit("DDD head never improved; aborting merge.")

    # Merge the best head into the full checkpoint.
    model.disc_head.load_state_dict(best_state)
    model.disc_head.eval()

    # Realistic inference uses the localizer's predicted centres rather than
    # the ground-truth labels, so report it alongside the teacher-forced score.
    realistic_metrics = evaluate(
        model, cache, val_cache_pos, device, class_weights,
        teacher_forced=False)

    checkpoint = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
    checkpoint["model_state_dict"] = model.state_dict()
    # Merge into the existing task flags; replacing the dict would drop
    # ``localization: true`` even though the localizer is exactly what we
    # just used (and the API reads these flags to report availability).
    checkpoint["tasks"] = merge_tasks(checkpoint, ddd=True)
    meta = dict(checkpoint.get("meta") or {})
    meta.update({
        "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": dict(checkpoint["tasks"]),
        "ddd_labels_used": True,
        "disc_pool_sigma_fraction": sigma_fraction,
        "ddd_selection": args.select_by,
        "disc_patch": {
            "dim": DISC_PATCH_DIM,
            "stats": DISC_PATCH_STATS,
            "half_w": DISC_PATCH_HALF_W,
            "half_h": DISC_PATCH_HALF_H,
            "res": list(DISC_PATCH_RES),
        },
        "ddd_finetune": {
            "cases": len(indices),
            "train_cases": len(train_idx),
            "val_cases": len(val_idx),
            "epochs_run": epoch,
            "best_epoch": best_epoch,
            "val_accuracy": round(best_metrics["accuracy"], 4),
            "val_within_one": round(best_metrics["within_one"], 4),
            "val_mae": round(best_metrics["mae"], 4),
            "val_teacher_forced_accuracy": round(
                best_metrics["accuracy"], 4),
            "val_predicted_centre_accuracy": round(
                realistic_metrics["accuracy"], 4),
            "val_predicted_centre_within_one": round(
                realistic_metrics["within_one"], 4),
            "val_discs": best_metrics["n"],
            "trainable_parameters": sum(
                p.numel() for p in model.disc_head.parameters()),
        },
    })
    checkpoint["meta"] = meta
    torch.save(checkpoint, args.output)

    print(f"\nSaved DDD checkpoint: {args.output}")
    if Path(args.output).resolve() == Path(BEST_MODEL).resolve():
        torch.save(checkpoint, LAST_MODEL)
        print(f"Also refreshed      : {LAST_MODEL}")
    print(f"Best val DDD accuracy: {best_metrics['accuracy']*100:.2f}% "
          f"(+/-1 grade {best_metrics['within_one']*100:.2f}%, "
          f"MAE {best_metrics['mae']:.3f} grades, "
          f"{best_metrics['n']} discs, epoch {best_epoch})")
    print(f"Val with predicted centres: acc "
          f"{realistic_metrics['accuracy']*100:.2f}% | "
          f"+/-1 {realistic_metrics['within_one']*100:.2f}% | "
          f"MAE {realistic_metrics['mae']:.3f}")
    print("\nNext: python evaluate.py --checkpoint "
          f"{args.output}")


if __name__ == "__main__":
    main()
