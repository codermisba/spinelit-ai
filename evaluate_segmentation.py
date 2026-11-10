"""
evaluate_segmentation.py
========================

Reproduce the held-out segmentation metrics for a trained U-Net checkpoint
and write visual overlays so the masks can be checked by eye.

The training script splits cases deterministically (case-level, 20% val,
seed 42).  This script rebuilds **exactly the same split** and reports
per-class Dice and IoU at the model resolution, plus mean foreground Dice —
the number stored in the checkpoint metadata.

Usage
-----
    python evaluate_segmentation.py
    python evaluate_segmentation.py --limit 10 --save-overlays 4

Outputs
-------
* console table of per-class Dice / IoU / pixel accuracy,
* ``outputs/segmentation/<case>_overlay.png`` (image | ground truth | prediction)
  for the first ``--save-overlays`` validation cases (default 4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config as _cfg
from segmentation import load_segmentation_model
from train_segmentation import FolderSegDataset, split_stems


DEFAULT_IMAGES = _cfg.DATA_DIR.parent / "segmentation" / "images"
DEFAULT_MASKS = _cfg.DATA_DIR.parent / "segmentation" / "masks"


def _colourise(mask: np.ndarray) -> np.ndarray:
    names = list(_cfg.SEG_CLASS_NAMES)
    palette = np.zeros((max(4, int(mask.max()) + 1), 3), dtype=np.uint8)
    for name, rgb in _cfg.SEG_CLASS_COLOURS.items():
        if name in names and names.index(name) < len(palette):
            palette[names.index(name)] = rgb
    return palette[mask]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(_cfg.SEG_CHECKPOINT))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    ap.add_argument("--masks", default=str(DEFAULT_MASKS))
    ap.add_argument("--image-size", type=int, default=None,
                    help="Defaults to the checkpoint's training size.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Evaluate only the first N validation cases.")
    ap.add_argument("--save-overlays", type=int, default=4,
                    help="Write overlay PNGs for this many validation cases.")
    ap.add_argument("--out-dir", default="outputs/segmentation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device, ckpt, meta = load_segmentation_model(args.checkpoint,
                                                        device)
    num_classes = int(meta["num_classes"])
    image_size = int(args.image_size or meta["image_size"])

    if not Path(args.images).is_dir() or not Path(args.masks).is_dir():
        raise SystemExit(
            f"Missing segmentation pairs. Build them first with:\n"
            f"  python prepare_segmentation_pairs.py\n"
            f"  (looked in {args.images} and {args.masks})"
        )

    full = FolderSegDataset(Path(args.images), Path(args.masks), image_size)
    _, val_stems = split_stems(full.stems, 0.2, 42)
    index_of = {stem: i for i, stem in enumerate(full.stems)}
    if args.limit:
        val_stems = val_stems[:args.limit]

    print(f"Checkpoint  : {ckpt}")
    print(f"Image size  : {image_size} px")
    print(f"Val cases   : {len(val_stems)} of {len(full.stems)} "
          f"(case-level 20% split, seed 42)")
    print(f"Device      : {device}")
    print()

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    denom = np.zeros(num_classes, dtype=np.float64)
    correct = total = 0

    overlay_dir = Path(args.out_dir)
    saved = 0

    for stem in val_stems:
        image_t, mask_t = full[index_of[stem]]
        with torch.no_grad():
            logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        target = mask_t.numpy().astype(np.uint8)

        for cls in range(num_classes):
            p = pred == cls
            t = target == cls
            inter[cls] += int((p & t).sum())
            union[cls] += int((p | t).sum())
            denom[cls] += int(p.sum()) + int(t.sum())
        correct += int((pred == target).sum())
        total += int(target.size)

        if saved < args.save_overlays:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            mean = np.asarray(_cfg.IMAGENET_MEAN, dtype=np.float32)
            std = np.asarray(_cfg.IMAGENET_STD, dtype=np.float32)
            raw = image_t.permute(1, 2, 0).numpy() * std + mean
            rgb = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
            panels = [
                Image.fromarray(rgb),
                Image.fromarray(_colourise(target)),
                Image.fromarray(_colourise(pred)),
            ]
            sheet = Image.new("RGB", (panels[0].width * 3, panels[0].height))
            for i, panel in enumerate(panels):
                sheet.paste(panel, (i * panels[0].width, 0))
            out = overlay_dir / f"{stem}_overlay.png"
            sheet.save(out)
            print(f"overlay -> {out}   (image | ground truth | prediction)")
            saved += 1

    dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1e-9), 0.0)
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

    print()
    print("Class                Dice     IoU   Pixels")
    print("-" * 46)
    for cls, name in enumerate(_cfg.SEG_CLASS_NAMES[:num_classes]):
        print(f"{name:<18} {dice[cls]:6.3f}  {iou[cls]:6.3f}  "
              f"{int(denom[cls] / 2):7d}")
    print("-" * 46)
    print(f"Mean foreground Dice : {dice[1:].mean():.3f}")
    print(f"Pixel accuracy       : {correct / max(1, total):.3f}")


if __name__ == "__main__":
    main()
