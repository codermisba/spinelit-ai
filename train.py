"""
train.py
=========

Training script for the Spine Foundation Model.

Features
--------
- Automatic device selection (CUDA if available, otherwise CPU)
- Stratified Train/Validation split
- AdamW optimizer
- CosineAnnealing learning-rate scheduler
- Gradient clipping
- Coordinate MSE loss + confidence BCE loss
- Early stopping
- Checkpoint saving (best + latest)
- Resume training
- CSV logging

Usage
-----
python train.py                          # train (resumes from latest if present)
python train.py --epochs 20              # override the number of epochs
python train.py --max-batches 5          # limit batches per epoch (smoke test)
python train.py --resume                 # force resume from latest_model.pth
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
    CPU_NUM_THREADS,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    WEIGHT_DECAY,
)
from dataset import SpineDataset
from model import SpineFoundationModel

# Reduce CPU heating / throttling by limiting PyTorch threads
torch.set_num_threads(CPU_NUM_THREADS)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

BEST_MODEL = CHECKPOINT_DIR / "best_model.pth"
LATEST_MODEL = CHECKPOINT_DIR / "latest_model.pth"

LOG_FILE = LOG_DIR / "training_log.csv"

# ---------------------------------------------------------
# Device (CUDA if available, otherwise CPU)
# ---------------------------------------------------------

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

        self.model = SpineFoundationModel().to(DEVICE)

        self.coord_loss_fn = nn.MSELoss()
        self.conf_loss_fn = nn.BCELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
        )

        # Mixed precision only on CUDA; no-op on CPU
        self.amp_enabled = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.amp_enabled else "cpu",
            enabled=self.amp_enabled,
        )

        self.best_loss = float("inf")
        self.start_epoch = 0

        self.train_loader = None
        self.val_loader = None

        self.create_log_file()

        if resume:
            self.load_checkpoint()

    # ---------------------------------------------------------
    # Dataset Split (stratified by source)
    # ---------------------------------------------------------

    def prepare_dataloaders(self) -> None:

        print("\nPreparing Train / Validation Split...")

        labels = []

        for filename in self.dataset.image_names:

            rows = self.dataset.groups.get_group(filename)

            labels.append(rows.iloc[0]["source"])

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=42,
        )

        train_idx, val_idx = next(
            splitter.split(self.dataset.image_names, labels)
        )

        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"Training Images   : {len(train_dataset)}")
        print(f"Validation Images : {len(val_dataset)}")

    # ---------------------------------------------------------
    # CSV Logger
    # ---------------------------------------------------------

    def create_log_file(self) -> None:

        if LOG_FILE.exists():
            return

        with open(LOG_FILE, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "Epoch",
                    "Train Loss",
                    "Validation Loss",
                    "Learning Rate",
                    "Timestamp",
                ]
            )

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
    ) -> None:

        with open(LOG_FILE, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    epoch,
                    round(train_loss, 6),
                    round(val_loss, 6),
                    self.optimizer.param_groups[0]["lr"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

    # ---------------------------------------------------------
    # Checkpoint Saving
    # ---------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
    ) -> None:

        checkpoint = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(checkpoint, LATEST_MODEL)

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            torch.save(checkpoint, BEST_MODEL)

            print("Best model updated.")    # ---------------------------------------------------------
    # Resume Training
    # ---------------------------------------------------------

    def load_checkpoint(self) -> None:

        if not LATEST_MODEL.exists():

            print("\nNo previous checkpoint found. Starting fresh.")

            return

        checkpoint = torch.load(
            LATEST_MODEL,
            map_location=DEVICE,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]

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

            if (
                self.max_batches is not None
                and batch_idx >= self.max_batches
            ):
                break

            images = batch["image"].to(DEVICE, non_blocking=True)
            coords = batch["coords"].to(DEVICE, non_blocking=True)
            confidence = batch["confidence"].to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda" if self.amp_enabled else "cpu",
                enabled=self.amp_enabled,
            ):

                outputs = self.model(images)

                coord_loss = self.coord_loss_fn(
                    outputs["coords"], coords
                )

            # BCELoss is not autocast-safe, so compute it in fp32
            conf_loss = self.conf_loss_fn(
                outputs["confidence"].float(), confidence
            )

            loss = coord_loss + (0.1 * conf_loss)

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=GRAD_CLIP_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            avg_loss = running_loss / (batch_idx + 1)

            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        num_batches = (
            self.max_batches
            if self.max_batches is not None
            else len(self.train_loader)
        )

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

            if (
                self.max_batches is not None
                and batch_idx >= self.max_batches
            ):
                break

            images = batch["image"].to(DEVICE, non_blocking=True)
            coords = batch["coords"].to(DEVICE, non_blocking=True)
            confidence = batch["confidence"].to(DEVICE, non_blocking=True)

            outputs = self.model(images)

            coord_loss = self.coord_loss_fn(outputs["coords"], coords)
            conf_loss = self.conf_loss_fn(outputs["confidence"], confidence)

            loss = coord_loss + (0.1 * conf_loss)

            running_loss += loss.item()

            avg_loss = running_loss / (batch_idx + 1)

            progress.set_postfix(val_loss=f"{avg_loss:.4f}")

        num_batches = (
            self.max_batches
            if self.max_batches is not None
            else len(self.val_loader)
        )

        return running_loss / num_batches

    # ---------------------------------------------------------
    # Main Training Loop
    # ---------------------------------------------------------

    def fit(self) -> None:

        self.prepare_dataloaders()

        print(f"\nUsing Device : {DEVICE}")
        print(f"Training for {self.epochs} epochs.")

        if self.start_epoch >= self.epochs:

            print(
                f"Model already trained for {self.epochs} epochs. "
                f"Nothing to do."
            )

            return

        patience_counter = 0

        for epoch in range(self.start_epoch, self.epochs):

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            self.scheduler.step()

            print(
                f"\nEpoch {epoch+1}/{self.epochs} "
                f"| Train Loss : {train_loss:.4f} "
                f"| Val Loss   : {val_loss:.4f}"
            )

            improved = val_loss < self.best_loss - 1e-5

            self.log_epoch(epoch + 1, train_loss, val_loss)

            self.save_checkpoint(epoch, val_loss)

            if improved:

                patience_counter = 0

            else:

                patience_counter += 1

                print(
                    f"No improvement for {patience_counter} "
                    f"epoch(s)."
                )

            if patience_counter >= EARLY_STOPPING_PATIENCE:

                print(
                    f"\nEarly stopping after {EARLY_STOPPING_PATIENCE} "
                    f"epochs without improvement."
                )

                break

        print("\nTraining finished.")
        print(f"Best validation loss : {self.best_loss:.6f}")
        print(f"Best model saved to  : {BEST_MODEL}")
        print(f"Latest model saved to: {LATEST_MODEL}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the Spine Foundation Model."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch (for quick smoke tests).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
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
