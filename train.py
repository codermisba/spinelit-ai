"""
train.py
========

Training script for the Spine Foundation Model (multi-task).

Tasks trained automatically based on label availability
--------------------------------------------------------
- Landmark localization      : always (disc CSV required)
- Localization confidence    : always (self-supervised)
- Disc degeneration (DDD)    : only when dataset/ddd_labels.csv exists
                               (per-level Pfirrmann multi-class 1-5)

Features
--------
- Automatic device selection (CUDA if available, otherwise CPU)
- Stratified Train/Validation split
- AdamW optimizer + CosineAnnealing LR scheduler
- Gradient clipping, mixed precision on GPU
- Early stopping, checkpoint saving (best + latest), resume, CSV logging

Usage
-----
python train.py                          # train (resumes from latest if present)
python train.py --epochs 20              # override the number of epochs
python train.py --max-batches 5          # quick smoke test
python train.py --resume                 # force resume from last_model.pth
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CONFIDENCE_SUPERVISION,
    CONFIDENCE_TEMPERATURE,
    COORD_LOSS_WEIGHT,
    CPU_NUM_THREADS,
    DDD_LOSS_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    NUM_DISCS,
    NUM_EPOCHS,
    NUM_KEYPOINTS,
    NUM_PFRRMANN_CLASSES,
    NUM_WORKERS,
    WEIGHT_DECAY,
    CONF_LOSS_WEIGHT,
    BEST_MODEL,
    LAST_MODEL,
    LOG_FILE,
)
from dataset import SpineDataset
from model import SpineFoundationModel

torch.set_num_threads(CPU_NUM_THREADS)

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(
        self,
        epochs: int = NUM_EPOCHS,
        max_batches: int | None = None,
        resume: bool = False,
    ):
        self.epochs = epochs
        self.max_batches = max_batches
        self.resume = resume

        print("\nLoading Dataset...")
        self.dataset = SpineDataset()
        print(f"Dataset Loaded  ->  Images : {len(self.dataset)}")
        print(f"DDD (Pfirrmann) labels available : {self.dataset.has_ddd_labels}")

        # DDD task enabled only when Pfirrmann labels exist
        self.train_ddd = self.dataset.has_ddd_labels and DDD_LOSS_WEIGHT > 0

        print(f"Training tasks : localization"
              f"{' + DDD (Pfirrmann)' if self.train_ddd else ''}")

        self.model = SpineFoundationModel().to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs,
        )

        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.amp_enabled else "cpu",
            enabled=self.amp_enabled,
        )

        self.best_loss = float("inf")
        self.start_epoch = 0

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

    def _masked_bce(self, pred, target, mask):
        if mask.sum() == 0:
            return pred.sum() * 0.0
        loss = nn.functional.binary_cross_entropy(
            pred.float(), target.float(), reduction="none"
        )
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def _masked_ce(self, logits, class_targets, mask):
        """Masked cross-entropy over the (B,5) per-disc class targets."""
        if mask.sum() == 0:
            return logits.sum() * 0.0
        B, D, C = logits.shape
        # Combine the batch and disc dims; keep mask per cell.
        flat_logits = logits.reshape(-1, C)
        flat_targets = class_targets.reshape(-1)
        flat_mask = mask.reshape(-1)
        ce = nn.functional.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        return (ce * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)

    # ---------------------------------------------------------
    # Dataset Split (stratified by source)
    # ---------------------------------------------------------

    def prepare_dataloaders(self) -> None:

        print("\nPreparing Train / Validation Split...")

        # Stratify by the scan "source" when present; otherwise every file is
        # its own group (e.g. SPIDER coords_pretrain.csv has no source column).
        first_df = self.dataset.disc_csv
        has_source = "source" in first_df.columns
        if has_source:
            labels = [
                self.dataset.groups.get_group(fn).iloc[0]["source"]
                for fn in self.dataset.image_names
            ]
        else:
            labels = list(range(len(self.dataset.image_names)))

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.20, random_state=42,
        )
        train_idx, val_idx = next(
            splitter.split(self.dataset.image_names, labels)
        )

        self.train_loader = DataLoader(
            Subset(self.dataset, train_idx),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = DataLoader(
            Subset(self.dataset, val_idx),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"Training Images   : {len(train_idx)}")
        print(f"Validation Images : {len(val_idx)}")

    # ---------------------------------------------------------
    # Batch loss (shared by train + validation)
    # ---------------------------------------------------------

    def compute_losses(self, batch, outputs):
        # Anchor every target to the device where the model OUTPUTS live
        # (images may still be on CPU when this is called).
        device = outputs["coords"].device

        coords = batch["coords"].to(device, non_blocking=True)
        visible = batch["point_visible"].to(device, non_blocking=True)

        coord_mask = visible.repeat_interleave(2, dim=1).clamp(0.0, 1.0)

        coord_diff = (outputs["coords"].float() - coords) * coord_mask
        denom = coord_mask.sum().clamp(min=2.0)
        coord_loss = (coord_diff ** 2).sum() / denom

        loc_conf_target = self._localization_confidence_target(coords, coords)
        loc_conf_loss = self._masked_bce(
            outputs["localization_conf"], loc_conf_target, visible
        )

        loss = (
            COORD_LOSS_WEIGHT * coord_loss
            + CONF_LOSS_WEIGHT * loc_conf_loss
        )

        task_losses = {"coord": coord_loss.item(), "loc_conf": loc_conf_loss.item()}

        if self.train_ddd:
            ddd_class = batch["ddd_class"].to(device, non_blocking=True)
            ddd_mask = batch["ddd_mask"].to(device, non_blocking=True)

            ddd_loss = self._masked_ce(
                outputs["ddd_logits"], ddd_class, ddd_mask
            )
            loss = loss + DDD_LOSS_WEIGHT * ddd_loss
            task_losses["ddd"] = ddd_loss.item()

        return loss, task_losses

    # ---------------------------------------------------------
    # CSV Logger
    # ---------------------------------------------------------

    def _create_log_file(self) -> None:

        if LOG_FILE.exists():
            return

        LOG_DIR.mkdir(exist_ok=True)

        with open(LOG_FILE, "w", newline="") as file:
            csv.writer(file).writerow(
                ["Epoch", "Train Loss", "Validation Loss", "Learning Rate",
                 "Timestamp"]
            )

    def log_epoch(self, epoch, train_loss, val_loss) -> None:

        with open(LOG_FILE, "a", newline="") as file:
            csv.writer(file).writerow(
                [epoch, round(train_loss, 6), round(val_loss, 6),
                 self.optimizer.param_groups[0]["lr"],
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            )

    # ---------------------------------------------------------
    # Checkpoints
    # ---------------------------------------------------------

    def save_checkpoint(self, epoch, val_loss) -> None:

        checkpoint = {
            "epoch": epoch,
            "best_loss": min(val_loss, self.best_loss),
            "model_config": {
                "num_keypoints": NUM_KEYPOINTS,
                "num_discs": NUM_DISCS,
            },
            "tasks": {
                "ddd": self.train_ddd,
            },
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(checkpoint, LAST_MODEL)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(checkpoint, BEST_MODEL)
            print("Best model updated.")

    def load_checkpoint(self) -> None:

        if not LAST_MODEL.exists():
            print("\nNo previous checkpoint found. Starting fresh.")
            return

        checkpoint = torch.load(
            LAST_MODEL, map_location=DEVICE, weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        print(f"\nResuming from epoch {self.start_epoch}")

    # ---------------------------------------------------------
    # Train One Epoch
    # ---------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> float:

        self.model.train()

        running_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):

            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            images = batch["image"].to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda" if self.amp_enabled else "cpu",
                enabled=self.amp_enabled,
            ):
                # Forward pass ONLY. Every loss below is computed outside
                # autocast in fp32 — BCELoss raises if autocast is active.
                outputs = self.model(images)

            loss, _ = self.compute_losses(batch, outputs)

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=GRAD_CLIP_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{running_loss/(batch_idx+1):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        num_batches = self.max_batches or len(self.train_loader)
        return running_loss / num_batches

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> float:

        self.model.eval()

        running_loss = 0.0

        progress = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{self.epochs} [Val]",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress):

            if self.max_batches is not None and batch_idx >= self.max_batches:
                break

            images = batch["image"].to(DEVICE, non_blocking=True)

            outputs = self.model(images)
            loss, _ = self.compute_losses(batch, outputs)

            running_loss += loss.item()

            progress.set_postfix(val_loss=f"{running_loss/(batch_idx+1):.4f}")

        num_batches = self.max_batches or len(self.val_loader)
        return running_loss / num_batches

    # ---------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------

    def fit(self) -> None:

        self.prepare_dataloaders()

        print(f"\nUsing Device : {DEVICE}")
        print(f"Training for {self.epochs} epochs.")

        if self.start_epoch >= self.epochs:
            print(f"Model already trained for {self.epochs} epochs.")
            return

        patience_counter = 0

        for epoch in range(self.start_epoch, self.epochs):

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step()

            improved = val_loss < self.best_loss - 1e-5

            print(
                f"\nEpoch {epoch+1}/{self.epochs}"
                f" | Train Loss : {train_loss:.4f}"
                f" | Val Loss   : {val_loss:.4f}"
            )

            self.log_epoch(epoch + 1, train_loss, val_loss)
            self.save_checkpoint(epoch, val_loss)

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s).")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping after {EARLY_STOPPING_PATIENCE}"
                    f" epochs without improvement."
                )
                break

        print("\nTraining finished.")
        print(f"Best validation loss : {self.best_loss:.6f}")
        print(f"Best model saved to  : {BEST_MODEL}")
        print(f"Latest model saved to: {LAST_MODEL}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the Spine Foundation Model."
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit batches per epoch (smoke test).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Force resume from the latest checkpoint.",
    )
    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs or NUM_EPOCHS,
        max_batches=args.max_batches,
        resume=args.resume,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
