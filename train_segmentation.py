"""
train_segmentation.py
=====================

Train the standalone 2-D U-Net segmentation model used by the API to draw
**true** vertebra / disc / spinal-canal masks on sagittal lumbar MRI.

Two supported data layouts
--------------------------
1. Simple image/mask folders (JPG/PNG, matched by filename stem):

       dataset/segmentation/images/<case>.jpg
       dataset/segmentation/masks/<case>.png

   Masks are single-channel class-index images (0 background, 1 vertebra,
   2 disc, 3 spinal canal).

2. nnU-Net raw layout produced by ``nnunet_prep.py``:

       nnUNet_raw/Dataset501_SPIDERMRI/imagesTr/<case>_0000.nii.gz
       nnUNet_raw/Dataset501_SPIDERMRI/labelsTr/<case>.nii.gz

   The mid-sagittal depth slice is extracted (the slice with the most
   foreground mask signal), matching the localizer's input view.

Loss / optimisation
-------------------
Soft Dice + cross-entropy, AdamW + cosine schedule, gradient clipping,
mixed precision on CUDA only.  Everything is CPU-feasible (small U-Net,
``--image-size 256 --batch-size 4``).

Usage
-----
    # real masks (folder layout)
    python train_segmentation.py --images dataset/segmentation/images \
        --masks dataset/segmentation/masks --epochs 30

    # nnU-Net layout
    python train_segmentation.py --nnunet_dir nnUNet_raw/Dataset501_SPIDERMRI \
        --epochs 30 --image-size 256 --batch-size 4

    # no data yet: verify the whole loop on synthetic blobs
    python train_segmentation.py --synthetic --epochs 2 --max-batches 5
"""

from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config as _cfg
from config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_SEG_CLASSES,
    SEG_CHECKPOINT,
)
from segmentation import UNet

CHECKPOINT_PATH = Path(SEG_CHECKPOINT)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

class FolderSegDataset(Dataset):
    """Image/mask folders with matching filename stems."""

    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int,
                 stems: list[str] | None = None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        available = {
            p.stem: p for p in sorted(self.image_dir.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif",
                                    ".tiff")
        }
        masks = {
            p.stem: p for p in sorted(self.mask_dir.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif",
                                    ".tiff")
        }
        self.pairs = [(stem, available[stem], masks[stem])
                      for stem in sorted(available) if stem in masks]
        if stems is not None:
            keep = set(stems)
            self.pairs = [p for p in self.pairs if p[0] in keep]
        if not self.pairs:
            raise SystemExit(
                f"No matched image/mask pairs between {self.image_dir} and "
                f"{self.mask_dir}."
            )

    @property
    def stems(self) -> list[str]:
        return [p[0] for p in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        _, image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return _resize_pair(image, mask, self.image_size)


class NnUNet2DDataset(Dataset):
    """Mid-sagittal 2-D slices from an nnU-Net raw dataset."""

    def __init__(self, dataset_dir: Path, image_size: int,
                 stems: list[str] | None = None):
        import SimpleITK as sitk  # noqa: F401  (fail early with clear error)

        self.dataset_dir = Path(dataset_dir)
        self.image_size = int(image_size)
        images_tr = self.dataset_dir / "imagesTr"
        labels_tr = self.dataset_dir / "labelsTr"
        if not images_tr.exists() or not labels_tr.exists():
            raise SystemExit(
                f"{self.dataset_dir} must contain imagesTr/ and labelsTr/."
            )
        self.pairs = []
        for image_path in sorted(images_tr.glob("*_0000.nii.gz")):
            stem = image_path.name.replace("_0000.nii.gz", "")
            label_path = labels_tr / f"{stem}.nii.gz"
            if label_path.exists():
                self.pairs.append((stem, image_path, label_path))
        if stems is not None:
            keep = set(stems)
            self.pairs = [p for p in self.pairs if p[0] in keep]
        if not self.pairs:
            raise SystemExit(f"No image/label pairs in {self.dataset_dir}.")

    @property
    def stems(self) -> list[str]:
        return [p[0] for p in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import SimpleITK as sitk

        _, image_path, label_path = self.pairs[index]
        image_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path)))

        signal = (mask_vol > 0).sum(axis=(0, 1))
        depth = int(np.argmax(signal)) if signal.max() > 0 \
            else mask_vol.shape[2] // 2

        image_slice = _to_uint8(image_vol[:, :, depth])
        mask_slice = mask_vol[:, :, depth].astype(np.uint8)
        image = Image.fromarray(image_slice).convert("RGB")
        mask = Image.fromarray(mask_slice, mode="L")
        return _resize_pair(image, mask, self.image_size)


class SyntheticSegDataset(Dataset):
    """Random MRI-like blobs + matching class masks (CPU smoke test)."""

    def __init__(self, length: int = 8, image_size: int = 128):
        self.length = int(length)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(index)
        s = self.image_size
        image = rng.normal(35, 20, (s, s)).astype(np.float32)
        mask = np.zeros((s, s), dtype=np.uint8)

        cx = s // 2 + rng.integers(-s // 10, s // 10)
        for i in range(5):
            y0 = int(s * (0.12 + i * 0.15))
            y1 = y0 + int(s * 0.09)
            x0 = cx - int(s * 0.11)
            x1 = cx + int(s * 0.11)
            image[y0:y1, x0:x1] += 90
            mask[y0:y1, x0:x1] = 1
            if i < 4:
                dy0, dy1 = y1, y1 + int(s * 0.035)
                mask[dy0:dy1, x0:x1] = 2
                image[dy0:dy1, x0:x1] += 40
        cy0, cy1 = int(s * 0.15), int(s * 0.8)
        cx0, cx1 = cx + int(s * 0.13), cx + int(s * 0.19)
        mask[cy0:cy1, cx0:cx1] = 3

        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        return _resize_pair(image, Image.fromarray(mask, mode="L"),
                            self.image_size)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return (arr * 255).astype(np.uint8)


def _resize_pair(image: Image.Image, mask: Image.Image, size: int):
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    mask = mask.resize((size, size), Image.NEAREST)

    img = np.asarray(image, dtype=np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.max() > NUM_SEG_CLASSES - 1:
        # Tolerate 0/255 binary masks by folding everything into "vertebra".
        unique = np.unique(mask_arr)
        if len(unique) <= 2:
            mask_arr = (mask_arr > 0).astype(np.uint8)
        else:
            mask_arr = np.clip(mask_arr, 0, NUM_SEG_CLASSES - 1)
    return img, torch.from_numpy(mask_arr.astype(np.int64))


# ----------------------------------------------------------------------
# Loss / metrics
# ----------------------------------------------------------------------

def split_stems(stems: list[str], val_fraction: float = 0.2,
                seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministic case-level train/val split (shared with evaluation)."""
    rng = np.random.default_rng(seed)
    order = list(stems)
    rng.shuffle(order)
    n_val = max(1, int(round(val_fraction * len(order))))
    return order[n_val:], order[:n_val]


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              num_classes: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dims)
    denom = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    # Ignore background for the loss; keep it in metrics.
    return 1.0 - dice[1:].mean()


def ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, label_smoothing=0.05)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

class SegTrainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = int(args.image_size)
        self.batch_size = int(args.batch_size)
        self.num_classes = int(NUM_SEG_CLASSES)
        self.epochs = int(args.epochs)

        self.train_set, self.val_set = self._build_datasets()
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=args.workers, drop_last=len(self.train_set) > self.batch_size,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=args.workers,
        )

        self.model = UNet(
            num_classes=self.num_classes,
            base_channels=args.base_channels,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs), eta_min=args.lr * 0.02,
        )
        self.amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.amp else "cpu", enabled=self.amp
        )
        self.best_dice = -1.0

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Segmentation model : U-Net ({n_params/1e6:.2f}M params)")
        print(f"Device             : {self.device}")
        print(f"Train / Val cases  : {len(self.train_set)} / {len(self.val_set)}")

    def _build_datasets(self):
        args = self.args
        if args.synthetic:
            full = SyntheticSegDataset(length=12, image_size=self.image_size)
            n_val = 3
            stems = list(range(len(full)))
            return (
                torch.utils.data.Subset(full, stems[n_val:]),
                torch.utils.data.Subset(full, stems[:n_val]),
            )

        if args.nnunet_dir:
            full = NnUNet2DDataset(Path(args.nnunet_dir), self.image_size)
        elif args.images and args.masks:
            full = FolderSegDataset(
                Path(args.images), Path(args.masks), self.image_size
            )
        else:
            raise SystemExit(
                "Provide --images/--masks, --nnunet_dir, or --synthetic."
            )

        stems = full.stems
        if len(stems) < 4:
            raise SystemExit(
                f"Only {len(stems)} cases; need at least 4 to split."
            )
        train_stems, val_stems = split_stems(stems, 0.2, 42)

        if isinstance(full, NnUNet2DDataset):
            train = NnUNet2DDataset(Path(args.nnunet_dir), self.image_size,
                                    stems=train_stems)
            val = NnUNet2DDataset(Path(args.nnunet_dir), self.image_size,
                                  stems=val_stems)
        else:
            train = FolderSegDataset(Path(args.images), Path(args.masks),
                                     self.image_size, stems=train_stems)
            val = FolderSegDataset(Path(args.images), Path(args.masks),
                                   self.image_size, stems=val_stems)
        return train, val

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            if self.args.max_batches and batch_idx >= self.args.max_batches:
                break
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                self.device.type, enabled=self.amp, dtype=torch.float16
            ):
                logits = self.model(images)
            logits = logits.float()
            loss = ce_loss(logits, masks) + dice_loss(
                logits, masks, self.num_classes
            )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item()
            n += 1
            if batch_idx % 5 == 0:
                print(f"  epoch {epoch+1} batch {batch_idx} "
                      f"loss={total/max(n,1):.4f}")
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self) -> tuple[float, np.ndarray]:
        self.model.eval()
        total, n = 0.0, 0
        inter = np.zeros(self.num_classes, dtype=np.float64)
        denom = np.zeros(self.num_classes, dtype=np.float64)
        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            logits = self.model(images).float()
            loss = ce_loss(logits, masks) + dice_loss(
                logits, masks, self.num_classes
            )
            total += loss.item()
            n += 1
            # Hard (argmax) Dice accumulated over the whole validation set, not
            # a per-batch average: this is exactly what evaluate_segmentation.py
            # reports, so the checkpoint metadata and the reproduced number
            # agree.
            pred = logits.argmax(dim=1)
            for cls in range(self.num_classes):
                p = (pred == cls)
                t = (masks == cls)
                inter[cls] += float((p & t).sum().item())
                denom[cls] += float(p.sum().item() + t.sum().item())
        dice = np.where(denom > 0, 2.0 * inter / np.maximum(denom, 1e-9), 0.0)
        return total / max(n, 1), dice

    def save(self, epoch: int, val_loss: float, dice: np.ndarray) -> None:
        payload = {
            "epoch": epoch,
            "best_dice": float(dice[1:].mean()),
            "meta": {
                "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "image_size": self.image_size,
                "num_classes": self.num_classes,
                "base_channels": self.args.base_channels,
                "class_names": list(_cfg.SEG_CLASS_NAMES),
                "val_dice": float(dice[1:].mean()),
                "val_dice_per_class": [float(v) for v in dice],
            },
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, CHECKPOINT_PATH)
        print(f"  saved -> {CHECKPOINT_PATH} (mean foreground Dice "
              f"{dice[1:].mean():.3f})")

    def fit(self) -> None:
        for epoch in range(self.epochs):
            started = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss, dice = self.validate()
            self.scheduler.step()
            print(
                f"Epoch {epoch+1}/{self.epochs} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | Dice "
                f"{' '.join(f'{v:.2f}' for v in dice)} | "
                f"{time.time()-started:.1f}s"
            )
            if dice[1:].mean() > self.best_dice:
                self.best_dice = float(dice[1:].mean())
                self.save(epoch + 1, val_loss, dice)
        print(f"Done. Best mean foreground Dice: {self.best_dice:.3f}")
        print(f"Checkpoint: {CHECKPOINT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the 2-D U-Net spine segmentation model."
    )
    ap.add_argument("--images", help="2-D image folder")
    ap.add_argument("--masks", help="2-D mask folder (class indices)")
    ap.add_argument("--nnunet_dir",
                    help="Dataset501_SPIDERMRI folder with imagesTr/labelsTr")
    ap.add_argument("--synthetic", action="store_true",
                    help="Train on synthetic blobs (smoke test only).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=_cfg.IMAGE_SIZE)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="Limit batches per epoch (0 = all).")
    args = ap.parse_args()

    set_seed(42)
    SegTrainer(args).fit()


if __name__ == "__main__":
    main()
