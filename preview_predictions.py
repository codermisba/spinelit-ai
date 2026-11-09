"""
preview_predictions.py
======================

Visual QA for a trained localizer: run the checkpoint on held-out MRI cases
and write side-by-side images showing

    predicted blocks/landmarks   |   ground-truth blocks/landmarks

plus a per-point pixel-error table.  This is the fastest way to confirm the
model really plots the right blocks on the MRI (rather than trusting a loss
number alone).

Usage
-----
python preview_predictions.py                       # 4 validation cases
python preview_predictions.py --num 8 --split train
python preview_predictions.py --checkpoint checkpoints/best_model.pth \
    --output outputs/predictions

Outputs land in ``outputs/predictions/`` (one PNG per case + ``summary.txt``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Subset

from config import DISC_LEVELS, ROOT_DIR, VERTEBRAE
from dataset import SpineDataset
from evaluate import build_val_indices
from utils import decode_outputs, load_model, render_analysis

POINT_NAMES = list(VERTEBRAE) + list(DISC_LEVELS)


def _font(size: int = 20):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _side_by_side(left: Image.Image, right: Image.Image,
                  left_label: str, right_label: str) -> Image.Image:
    gap, header = 14, 38
    width = max(left.width, right.width)
    height = max(left.height, right.height)
    canvas = Image.new(
        "RGB", (width * 2 + gap, height + header), (255, 255, 255)
    )
    font = _font(22)
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), left_label, fill=(20, 20, 20), font=font)
    draw.text((width + gap + 10, 8), right_label, fill=(20, 20, 20), font=font)
    canvas.paste(left, (0, header))
    canvas.paste(right, (width + gap, header))
    return canvas


def _select_indices(dataset: SpineDataset, split: str, num: int) -> list[int]:
    val_idx = np.asarray(build_val_indices(dataset), dtype=int)
    if split == "val":
        pool = val_idx
    elif split == "train":
        mask = np.ones(len(dataset), dtype=bool)
        mask[val_idx] = False
        pool = np.flatnonzero(mask)
    else:
        pool = np.arange(len(dataset))
    if len(pool) == 0:
        raise SystemExit(f"No images available for split={split!r}.")
    # Deterministic, evenly spread sample.
    picks = np.linspace(0, len(pool) - 1, min(num, len(pool))).round().astype(int)
    return [int(pool[i]) for i in picks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render predicted vs ground-truth spine blocks on MRI."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--split", choices=["val", "train", "all"],
                        default="val")
    parser.add_argument("--num", type=int, default=4,
                        help="Number of cases to render.")
    parser.add_argument("--output", default=str(ROOT_DIR / "outputs" / "predictions"))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"No checkpoint at {checkpoint_path}. Train first.")

    # load_model applies the checkpoint's own image_size/backbone/norm to
    # config, so the dataset below preprocesses identically.
    model, device, _, meta = load_model(str(checkpoint_path))
    image_size = int(meta.get("image_size") or 256)

    dataset = SpineDataset()
    indices = _select_indices(dataset, args.split, args.num)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    all_errors: list[float] = []
    visible_flags: list[bool] = []

    for rank, index in enumerate(indices, start=1):
        sample = dataset[index]
        filename = dataset.image_names[index]
        x = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x, return_seg=False)
        decoded = decode_outputs(outputs)

        pred = np.asarray(decoded["points"], dtype=np.float32).reshape(-1, 2)
        conf = np.asarray(decoded["localization_conf"],
                          dtype=np.float32).reshape(-1)
        gt = sample["coords"].numpy().reshape(-1, 2)
        visible = sample["point_visible"].numpy().reshape(-1) > 0.5

        # Same pixel convention as evaluate.py: normalized error * input size.
        errors = np.linalg.norm(pred - gt, axis=1) * image_size
        all_errors.append(errors[visible])
        visible_flags.append(visible)

        image_path = dataset._find_image(filename)
        pil = Image.open(image_path).convert("RGB")

        pred_image = render_analysis(
            pil, coords=pred, confidence=conf,
            show_blocks=True, show_landmarks=True,
        )
        gt_image = render_analysis(
            pil, coords=gt, confidence=visible.astype(np.float32),
            show_blocks=True, show_landmarks=True,
        )

        stem = Path(filename).stem
        pred_image.save(out_dir / f"{rank:02d}_{stem}_pred.png")
        body = _side_by_side(
            pred_image, gt_image,
            f"PREDICTION  (mean {errors[visible].mean():.2f} px)",
            "GROUND TRUTH",
        )
        body.save(out_dir / f"{rank:02d}_{stem}_compare.png")

        summary_lines.append(
            f"\n[{rank}] {filename}  "
            f"(mean {errors[visible].mean():.2f} px, "
            f"PCK@10% {(errors[visible] <= 0.10 * image_size).mean():.3f})"
        )
        for i, name in enumerate(POINT_NAMES):
            if not visible[i]:
                summary_lines.append(f"    {name:<6} (not annotated)")
                continue
            summary_lines.append(
                f"    {name:<6} {errors[i]:6.2f} px   conf {conf[i]:.3f}"
            )

    flat = np.concatenate(all_errors)
    flags = np.concatenate(visible_flags)
    summary_lines.append("\n" + "=" * 52)
    summary_lines.append(f"Cases              : {len(indices)} ({args.split})")
    summary_lines.append(f"Visible points     : {len(flat)}")
    summary_lines.append(f"Mean error (px)    : {flat.mean():.2f}")
    summary_lines.append(f"Median error (px)  : {np.median(flat):.2f}")
    summary_lines.append(
        f"PCK@5%             : {(flat <= 0.05 * image_size).mean() * 100:.2f} %"
    )
    summary_lines.append(
        f"PCK@10%            : {(flat <= 0.10 * image_size).mean() * 100:.2f} %"
    )
    summary_lines.append(f"Points annotated   : {int(flags.sum())}")
    summary_lines.append(f"Checkpoint         : {checkpoint_path}")
    summary_lines.append(f"Image size         : {image_size} px")
    summary_lines.append(
        f"DDD head trained   : {bool(meta.get('tasks', {}).get('ddd', False))}"
    )

    text = "\n".join(summary_lines)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nImages written to: {out_dir}")


if __name__ == "__main__":
    main()
