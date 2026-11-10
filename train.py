"""
train.py
========

Training script for the rebuilt Spine Foundation Model.

Tasks trained automatically based on label availability
-------------------------------------------------------
- Landmark localisation      : always (disc CSV required)
  * dense Gaussian heatmap per keypoint (main supervision)
  * differentiable soft-argmax coordinate loss
  * self-supervised per-point confidence
- Disc degeneration (DDD)    : only when dataset/ddd_labels.csv exists
                               (per-level Pfirrmann 1-5)
- Segmentation               : only when the dataset yields masks
                               (see train_segmentation.py for the
                               dedicated nnU-Net-style mask trainer)

Fixes vs. the previous version
------------------------------
* Confidence target now uses the model prediction (was `coords, coords`,
  so it always targeted 1.0).
* DDD pooling uses ground-truth disc centres while training (teacher
  forcing) instead of random early-training predictions.
* ImageNet normalisation is applied consistently (matches inference and
  the pretrained backbone).
* Losses are computed in fp32 outside autocast (BCE-safe).
* Deterministic ordering / seeding, correct per-epoch averaging,
  resume-safe scheduler, richer checkpoint metadata.

Usage
-----
python train.py                              # full run
python train.py --model convnext_femto --image-size 192 --batch-size 16
python train.py --epochs 3 --max-batches 5   # smoke test
python train.py --resume                     # continue from last_model.pth
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config as _cfg
from config import (
    BATCH_SIZE,
    BEST_MODEL,
    CONF_LOSS_WEIGHT,
    CONFIDENCE_SUPERVISION,
    CONFIDENCE_TEMPERATURE,
    COORD_LOSS_WEIGHT,
    CPU_NUM_THREADS,
    DDD_LABEL_SMOOTHING,
    DDD_LOSS_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    HEATMAP_LOSS_WEIGHT,
    HEATMAP_SIGMA,
    LAST_MODEL,
    LEARNING_RATE,
    LOCALIZATION_SELECTION_WEIGHT,
    LOG_FILE,
    MODEL_NAME,
    NUM_DISCS,
    NUM_EPOCHS,
    NUM_KEYPOINTS,
    NUM_PFRRMANN_CLASSES,
    NUM_VERTEBRAE,
    NUM_WORKERS,
    SEG_LOSS_WEIGHT,
    WEIGHT_DECAY,
)
from dataset import SpineDataset
from model import SpineFoundationModel

torch.set_num_threads(CPU_NUM_THREADS)

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_HEADER = [
    "epoch", "train_loss", "val_loss", "coord_loss", "heatmap_loss",
    "conf_loss", "ddd_loss", "seg_loss", "kp_error_px", "pck@5",
    "pck@10", "lr", "timestamp",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shrink_rare_labels(labels):
    """Collapse classes with <2 members so StratifiedShuffleSplit works."""
    counts = Counter(labels)
    rare = {lab for lab, n in counts.items() if n < 2}
    if not rare:
        return list(labels)
    return [("other" if lab in rare else lab) for lab in labels]


def gaussian_heatmap_targets(
    coords: torch.Tensor,
    visibility: torch.Tensor,
    hf: int,
    wf: int,
    sigma: float = HEATMAP_SIGMA,
) -> torch.Tensor:
    """Build (B,K,hf,wf) Gaussian heatmap targets from normalized coords."""
    b = coords.shape[0]
    pts = coords.float().view(b, NUM_KEYPOINTS, 2)
    cx = pts[:, :, 0].clamp(0.0, 1.0) * (wf - 1)
    cy = pts[:, :, 1].clamp(0.0, 1.0) * (hf - 1)

    yy = torch.arange(hf, device=coords.device, dtype=torch.float32).view(
        1, 1, hf, 1)
    xx = torch.arange(wf, device=coords.device, dtype=torch.float32).view(
        1, 1, 1, wf)
    d2 = (xx - cx.view(b, NUM_KEYPOINTS, 1, 1)).square() + \
         (yy - cy.view(b, NUM_KEYPOINTS, 1, 1)).square()
    target = torch.exp(-d2 / (2.0 * sigma * sigma))
    return target * visibility.view(b, NUM_KEYPOINTS, 1, 1)


def focal_heatmap_loss(logits, target, visibility):
    """Foreground-weighted BCE (background dominates heatmaps)."""
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    weight = 1.0 + 50.0 * target
    weight = weight * visibility.view(
        visibility.shape[0], NUM_KEYPOINTS, 1, 1)
    denom = weight.sum().clamp(min=1.0)
    return (bce * weight).sum() / denom


class Trainer:

    def __init__(
        self,
        epochs: int = NUM_EPOCHS,
        max_batches: int | None = None,
        resume: bool = False,
        batch_size: int = BATCH_SIZE,
        image_size: int | None = None,
        model_name: str | None = None,
        lr: float = LEARNING_RATE,
        limit: int | None = None,
        workers: int = NUM_WORKERS,
        pretrained: bool = True,
    ):
        self.epochs = epochs
        self.max_batches = max_batches
        self.resume = resume
        self.batch_size = batch_size
        self.workers = workers

        # Checkpoint meta can override the architecture size before we
        # construct the network (resume with a model trained at another
        # resolution / backbone).
        if resume and LAST_MODEL.exists():
            ckpt = torch.load(LAST_MODEL, map_location="cpu", weights_only=False)
            from utils import _apply_checkpoint_meta
            _apply_checkpoint_meta(ckpt.get("meta") or {})

        # An explicitly supplied image size wins; otherwise a resumed
        # checkpoint keeps the resolution it was trained at (256 px default
        # for a fresh run).
        if image_size is not None:
            _cfg.IMAGE_SIZE = int(image_size)
        if model_name:
            _cfg.MODEL_NAME = model_name
        _cfg.PRETRAINED = bool(pretrained)

        print("\nLoading Dataset...")
        self.dataset = SpineDataset()
        print(f"Dataset Loaded  ->  Images : {len(self.dataset)}")

        # Same ordering, training view adds horizontal-flip augmentation.
        self.train_dataset = SpineDataset(augment=True)
        if self.train_dataset.image_names != self.dataset.image_names:
            raise RuntimeError(
                "Augmented train dataset must use the same image ordering."
            )
        print("Data augmentation : horizontal flip (training split only)")

        n_on_disk = sum(
            1 for fn in self.dataset.image_names
            if any((d / fn).exists() for d in self.dataset.image_dirs)
        )
        if n_on_disk < len(self.dataset.image_names):
            print(
                f"[warn] Only {n_on_disk}/{len(self.dataset.image_names)} "
                "annotated images were found on disk."
            )
        print(f"DDD (Pfirrmann) labels available : {self.dataset.has_ddd_labels}")

        self.train_ddd = self.dataset.has_ddd_labels and DDD_LOSS_WEIGHT > 0
        # The dataset currently does not yield masks; the dedicated
        # train_segmentation.py handles the mask task.  This flag keeps
        # the loss code ready for a future joint dataset.
        self.train_seg = False

        print(
            "Training tasks : localisation"
            f"{' + DDD (Pfirrmann)' if self.train_ddd else ''}"
        )
        if not self.train_ddd:
            print(
                "[info] No dataset/ddd_labels.csv -> the DDD head is NOT "
                "trained. The API will report DDD grading as unavailable "
                "instead of returning random grades."
            )

        self.model = SpineFoundationModel().to(DEVICE)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model           : {_cfg.MODEL_NAME} ({n_params/1e6:.1f}M params)")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs), eta_min=lr * 0.02,
        )

        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.amp_enabled else "cpu",
            enabled=self.amp_enabled,
        )

        self.best_loss = float("inf")
        self.best_metric = float("inf")
        self.start_epoch = 0
        self.limit = limit
        self.train_loader = None
        self.val_loader = None

        self._create_log_file()
        if resume:
            self.load_checkpoint()

    # ---------------------------------------------------------
    # Self-supervised confidence targets
    # ---------------------------------------------------------

    def _localization_confidence_target(self, coords_pred, coords_target):
        """
        confidence_target = exp(-euclidean_error / TEMPERATURE) per point,
        computed in fp32 (BCE is not autocast-safe).
        """
        pred = coords_pred.float().view(-1, NUM_KEYPOINTS, 2)
        target = coords_target.float().view(-1, NUM_KEYPOINTS, 2)
        per_point_dist = torch.sqrt(
            ((pred - target) ** 2).sum(dim=2) + 1e-12
        )
        if not CONFIDENCE_SUPERVISION:
            return torch.ones_like(per_point_dist)
        return torch.exp(-per_point_dist / CONFIDENCE_TEMPERATURE)

    # ---------------------------------------------------------
    # Masked losses
    # ---------------------------------------------------------

    @staticmethod
    def _masked_bce(pred, target, mask):
        if mask.sum() == 0:
            return pred.sum() * 0.0
        loss = F.binary_cross_entropy(
            pred.float(), target.float(), reduction="none"
        )
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _masked_ce(logits, class_targets, mask):
        if mask.sum() == 0:
            return logits.sum() * 0.0
        c = logits.shape[-1]
        flat_logits = logits.reshape(-1, c)
        flat_targets = class_targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        ce = F.cross_entropy(
            flat_logits, flat_targets, reduction="none",
            label_smoothing=DDD_LABEL_SMOOTHING,
        )
        return (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

    # ---------------------------------------------------------
    # Dataset split
    # ---------------------------------------------------------

    def _split_labels(self) -> list:
        if "source" in self.dataset.disc_csv.columns:
            labels = [
                self.dataset.groups.get_group(fn).iloc[0]["source"]
                for fn in self.dataset.image_names
            ]
            return _shrink_rare_labels(labels)

        ddd = self.dataset.ddd_labels
        if ddd is not None and len(ddd):
            labels = []
            for fn in self.dataset.image_names:
                rows = ddd[ddd["filename"].astype(str) == str(fn)]
                if len(rows) == 0:
                    labels.append("nograde")
                else:
                    g = rows["pfirrmann_grade"].astype(float)
                    labels.append(str(int(round(g.median()))))
            return _shrink_rare_labels(labels)

        raise ValueError(
            "Cannot stratify split: no 'source' column and no DDD labels."
        )

    def prepare_dataloaders(self) -> None:
        print("\nPreparing Train / Validation Split...")
        n_images = len(self.dataset.image_names)
        if n_images < 10:
            raise SystemExit(
                f"Only {n_images} unique images in coords_pretrain.csv - too "
                "few to train/split. Regenerate it with prepare_spider.py or "
                "the dataset builder."
            )

        labels = self._split_labels()
        n_classes = len(set(labels))
        n_val = max(1, int(round(0.15 * n_images)))

        if n_images >= 20 and n_val >= n_classes:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.15, random_state=42,
            )
            train_idx, val_idx = next(
                splitter.split(self.dataset.image_names, labels)
            )
        else:
            print(f"[warn] Using random split ({n_images} images).")
            perm = torch.randperm(
                n_images, generator=torch.Generator().manual_seed(42)
            ).tolist()
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

        if self.limit:
            train_idx = train_idx[: self.limit]
            val_idx = val_idx[: max(1, self.limit // 4)]

        self.train_loader = DataLoader(
            Subset(self.train_dataset, train_idx),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(train_idx) > self.batch_size,
        )
        self.val_loader = DataLoader(
            Subset(self.dataset, val_idx),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
        )
        if len(self.train_loader) == 0:
            raise SystemExit(
                "Training loader is empty. Increase --limit or lower "
                "--batch-size."
            )
        print(f"Training Images   : {len(train_idx)}")
        print(f"Validation Images : {len(val_idx)}")

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def compute_losses(self, batch, outputs):
        device = outputs["coords"].device

        coords = batch["coords"].to(device, non_blocking=True)
        visible = batch["point_visible"].to(device, non_blocking=True)

        coord_mask = visible.repeat_interleave(2, dim=1).clamp(0.0, 1.0)
        coord_diff = (outputs["coords"].float() - coords) * coord_mask
        coord_loss = (coord_diff ** 2).sum() / coord_mask.sum().clamp(min=2.0)

        heatmaps = outputs["heatmaps"]
        _, _, hf, wf = heatmaps.shape
        target = gaussian_heatmap_targets(coords, visible, hf, wf)
        heatmap_loss = focal_heatmap_loss(heatmaps, target, visible)

        # -- FIX: use the model prediction here, not the target twice.
        conf_target = self._localization_confidence_target(
            outputs["coords"], coords)
        conf_loss = self._masked_bce(
            outputs["localization_conf"], conf_target, visible
        )

        loss = (
            COORD_LOSS_WEIGHT * coord_loss
            + HEATMAP_LOSS_WEIGHT * heatmap_loss
            + CONF_LOSS_WEIGHT * conf_loss
        )
        task = {
            "coord": coord_loss.item(),
            "heatmap": heatmap_loss.item(),
            "loc_conf": conf_loss.item(),
            "ddd": 0.0,
            "seg": 0.0,
        }

        if self.train_ddd:
            ddd_class = batch["ddd_class"].to(device, non_blocking=True)
            ddd_mask = batch["ddd_mask"].to(device, non_blocking=True)
            ddd_loss = self._masked_ce(
                outputs["ddd_logits"], ddd_class, ddd_mask
            )
            loss = loss + DDD_LOSS_WEIGHT * ddd_loss
            task["ddd"] = ddd_loss.item()

        if self.train_seg and "mask" in batch:
            mask = batch["mask"].to(device, non_blocking=True).long()
            seg_logits = F.interpolate(
                outputs["seg_logits"], size=mask.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            seg_loss = F.cross_entropy(seg_logits, mask)
            loss = loss + SEG_LOSS_WEIGHT * seg_loss
            task["seg"] = seg_loss.item()

        return loss, task

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    def _keypoint_metrics(self, outputs, batch) -> tuple[float, float, float]:
        """Return (mean error in px, PCK@5%, PCK@10%) on visible points."""
        device = outputs["coords"].device
        coords = batch["coords"].to(device).float().view(-1, NUM_KEYPOINTS, 2)
        visible = batch["point_visible"].to(device).float()
        pred = outputs["coords"].float().view(-1, NUM_KEYPOINTS, 2)

        dist = torch.sqrt(((pred - coords) ** 2).sum(dim=-1) + 1e-12)
        mask = visible > 0.5
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        d = dist[mask]
        # `d` is in normalized image units, so PCK@5% = within 0.05.
        pck5 = (d <= 0.05).float().mean().item()
        pck10 = (d <= 0.10).float().mean().item()
        return float(d.mean().item()), pck5, pck10

    # ---------------------------------------------------------
    # Logging / checkpoints
    # ---------------------------------------------------------

    def _create_log_file(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        if not LOG_FILE.exists():
            with open(LOG_FILE, "w", newline="") as fh:
                csv.writer(fh).writerow(CSV_HEADER)

    def log_epoch(self, row) -> None:
        with open(LOG_FILE, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

    def _checkpoint_payload(self, epoch, val_loss, selection_metric) -> dict:
        tasks = {"ddd": self.train_ddd, "seg": self.train_seg}
        return {
            "epoch": epoch,
            "best_loss": min(val_loss, self.best_loss),
            "best_metric": min(selection_metric, self.best_metric),
            "selection_metric": selection_metric,
            "model_config": {
                "num_keypoints": NUM_KEYPOINTS,
                "num_discs": NUM_DISCS,
            },
            "tasks": tasks,
            "meta": {
                "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "image_size": _cfg.IMAGE_SIZE,
                "disc_feature_stage": _cfg.DISC_FEATURE_STAGE,
                "model_name": _cfg.MODEL_NAME,
                "fpn_dim": _cfg.FPN_DIM,
                "heatmap_stride": _cfg.HEATMAP_STRIDE,
                "softargmax_temperature": _cfg.SOFTARGMAX_TEMPERATURE,
                "disc_pool_sigma_fraction": _cfg.DISC_POOL_SIGMA_FRACTION,
                "num_keypoints": NUM_KEYPOINTS,
                "num_discs": NUM_DISCS,
                "num_pfrrmann_classes": NUM_PFRRMANN_CLASSES,
                "num_seg_classes": _cfg.NUM_SEG_CLASSES,
                "use_imagenet_norm": _cfg.USE_IMAGENET_NORM,
                "tasks": tasks,
                "ddd_labels_used": self.train_ddd,
            },
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def save_checkpoint(self, epoch, val_loss, selection_metric) -> None:
        payload = self._checkpoint_payload(epoch, val_loss, selection_metric)
        torch.save(payload, LAST_MODEL)
        if selection_metric < self.best_metric - 1e-6:
            self.best_metric = selection_metric
            self.best_loss = min(self.best_loss, val_loss)
            payload = self._checkpoint_payload(epoch, val_loss, selection_metric)
            torch.save(payload, BEST_MODEL)
            print(
                f"Best model updated (selection={selection_metric:.5f}, "
                f"val_loss={val_loss:.5f})."
            )

    def load_checkpoint(self) -> None:
        if not LAST_MODEL.exists():
            print("\nNo previous checkpoint found. Starting fresh.")
            return
        checkpoint = torch.load(
            LAST_MODEL, map_location=DEVICE, weights_only=False)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            print(f"\n[resume] incompatible checkpoint ({exc}). Starting fresh.")
            return
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        print(f"\nResuming from epoch {self.start_epoch}")

    # ---------------------------------------------------------
    # Epochs
    # ---------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> tuple[float, dict]:
        self.model.train()
        running = 0.0
        totals = Counter()
        n_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
            leave=False,
        )
        for batch_idx, batch in enumerate(progress):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            images = batch["image"].to(DEVICE, non_blocking=True)
            # Teacher forcing for the disc head: ground-truth disc centres
            # give a clean Pfirrmann feature patch during training.
            gt_centers = batch["coords"][:, NUM_VERTEBRAE * 2:].view(
                -1, NUM_DISCS, 2).to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda" if self.amp_enabled else "cpu",
                enabled=self.amp_enabled,
            ):
                outputs = self.model(images, disc_centers=gt_centers)

            loss, task = self.compute_losses(batch, outputs)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=GRAD_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item()
            totals.update(task)
            n_batches += 1
            progress.set_postfix(
                loss=f"{running/n_batches:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        n = max(1, n_batches)
        avg = {k: v / n for k, v in totals.items()}
        return running / n, avg

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[float, dict, dict]:
        self.model.eval()
        running = 0.0
        totals = Counter()
        n_batches = 0
        err_sum = 0.0
        pck5_sum = 0.0
        pck10_sum = 0.0
        n_metric = 0

        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
            leave=False,
        )
        for batch_idx, batch in enumerate(progress):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            images = batch["image"].to(DEVICE, non_blocking=True)
            gt_centers = batch["coords"][:, NUM_VERTEBRAE * 2:].view(
                -1, NUM_DISCS, 2).to(DEVICE)
            outputs = self.model(images, disc_centers=gt_centers)
            loss, task = self.compute_losses(batch, outputs)

            running += loss.item()
            totals.update(task)
            n_batches += 1

            err, p5, p10 = self._keypoint_metrics(outputs, batch)
            if not math.isnan(err):
                err_sum += err
                pck5_sum += p5
                pck10_sum += p10
                n_metric += 1

            progress.set_postfix(val_loss=f"{running/n_batches:.4f}")

        n = max(1, n_batches)
        m = max(1, n_metric)
        metrics = {
            "kp_error_px": err_sum / m * _cfg.IMAGE_SIZE,
            "pck5": pck5_sum / m,
            "pck10": pck10_sum / m,
        }
        return running / n, {k: v / n for k, v in totals.items()}, metrics

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------

    def fit(self) -> None:
        self.prepare_dataloaders()

        print(f"\nUsing Device : {DEVICE}")
        print(f"Training for {self.epochs} epochs.")
        if self.start_epoch >= self.epochs:
            print(f"Model already trained for {self.epochs} epochs.")
            return

        patience = 0
        for epoch in range(self.start_epoch, self.epochs):
            train_loss, train_task = self.train_one_epoch(epoch)
            val_loss, val_task, metrics = self.validate(epoch)
            self.scheduler.step()

            selection_metric = val_loss + LOCALIZATION_SELECTION_WEIGHT * (
                metrics["kp_error_px"] / max(1, _cfg.IMAGE_SIZE)
            )
            improved = selection_metric < self.best_metric - 1e-6

            print(
                f"\nEpoch {epoch+1}/{self.epochs}"
                f" | Train {train_loss:.4f}"
                f" | Val {val_loss:.4f}"
                f" | KP err {metrics['kp_error_px']:.1f}px"
                f" | PCK@5% {metrics['pck5']:.3f}"
                f" | PCK@10% {metrics['pck10']:.3f}"
                f" | lr {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            self.log_epoch([
                epoch + 1,
                round(train_loss, 6),
                round(val_loss, 6),
                round(train_task["coord"], 6),
                round(train_task["heatmap"], 6),
                round(train_task["loc_conf"], 6),
                round(train_task["ddd"], 6),
                round(train_task["seg"], 6),
                round(metrics["kp_error_px"], 3),
                round(metrics["pck5"], 5),
                round(metrics["pck10"], 5),
                self.optimizer.param_groups[0]["lr"],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])
            self.save_checkpoint(epoch, val_loss, selection_metric)

            if improved:
                patience = 0
            else:
                patience += 1
                print(f"No improvement for {patience} epoch(s).")
            if patience >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping after {EARLY_STOPPING_PATIENCE} "
                    "epochs without improvement."
                )
                break

        print("\nTraining finished.")
        print(f"Best validation loss : {self.best_loss:.6f}")
        print(f"Best model saved to  : {BEST_MODEL}")
        print(f"Latest model saved to: {LAST_MODEL}")
        print("\nNext: python evaluate.py   (or start the API: "
              "uvicorn api_server:app --port 8000)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the rebuilt Spine Foundation Model."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch (smoke test).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/last_model.pth.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input resolution (default: config, or the "
                             "checkpoint's resolution when resuming).")
    parser.add_argument("--model", default=None,
                        help="timm backbone name (default: config.MODEL_NAME).")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use at most N training images (quick runs).")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download ImageNet weights.")
    args = parser.parse_args()

    set_seed(42)
    trainer = Trainer(
        epochs=args.epochs or NUM_EPOCHS,
        max_batches=args.max_batches,
        resume=args.resume,
        batch_size=args.batch_size,
        image_size=args.image_size,
        model_name=args.model,
        lr=args.lr,
        limit=args.limit,
        workers=args.workers,
        pretrained=not args.no_pretrained,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
