"""
prepare_segmentation_pairs.py
=============================

Build 2-D image/mask pairs for ``train_segmentation.py`` from the processed
SPIDER volumes and the extracted SPIDER ``.mha`` masks.

Inputs
------
    dataset/data/processed_spider/<case>.npy   (256,256,N) uint8 mid-sagittal stack
    dataset/data/spider_raw/masks/<case>.mha   (H,W,N) integer class mask

Outputs
-------
    dataset/segmentation/images/<case>.png
    dataset/segmentation/masks/<case>.png      (class indices 0..3)
    dataset/segmentation/manifest.csv

Class mapping (SPIDER legend -> pipeline classes)
-------------------------------------------------
    1..25, 101..125  -> 1 vertebra
    201..225         -> 2 disc
    100              -> 3 spinal canal
    0                -> 0 background

The sagittal depth is chosen exactly like ``prepare_spider.py``: the slice
with the most foreground mask signal. Cases whose mask depth does not match
the processed volume depth are skipped (a real geometry mismatch would teach
the network wrong anatomy), and the manifest records why.

Usage
-----
    python prepare_segmentation_pairs.py
    python prepare_segmentation_pairs.py --min-intensity-ratio 1.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

from config import DATA_DIR, NUM_SEG_CLASSES, ROOT_DIR

NPY_DIR = DATA_DIR / "processed_spider"
MASK_DIR = DATA_DIR / "spider_raw" / "masks"
OUT_DIR = ROOT_DIR / "dataset" / "segmentation"


def map_classes(mask: np.ndarray) -> np.ndarray:
    """Map raw SPIDER labels to 0 background / 1 vertebra / 2 disc / 3 canal."""
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[((mask >= 1) & (mask <= 25)) | ((mask >= 101) & (mask <= 125))] = 1
    out[(mask >= 201) & (mask <= 225)] = 2
    out[mask == 100] = 3
    return out


def best_depth(mask_vol: np.ndarray) -> int:
    signal = (mask_vol > 0).sum(axis=(0, 1))
    if signal.max() == 0:
        return -1
    return int(np.argmax(signal))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-dir", default=str(NPY_DIR))
    parser.add_argument("--mask-dir", default=str(MASK_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--min-intensity-ratio", type=float, default=1.02,
                        help="Skip pairs whose mask does not cover brighter "
                             "anatomy than the background.")
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    image_out = out_dir / "images"
    mask_out = out_dir / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # Start clean so a rerun cannot mix stale pairs with new filters.
    for folder in (image_out, mask_out):
        for old in folder.glob("*.png"):
            old.unlink()

    cases = sorted(p.stem for p in npy_dir.glob("*.npy"))
    rows = []
    for case in cases:
        mask_path = mask_dir / f"{case}.mha"
        if not mask_path.exists():
            rows.append({"case": case, "status": "no_mask"})
            continue
        volume = np.load(npy_dir / f"{case}.npy")
        mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        if mask_vol.shape[2] != volume.shape[2]:
            rows.append({"case": case, "status": "depth_mismatch",
                         "npy_depth": volume.shape[2],
                         "mask_depth": mask_vol.shape[2]})
            continue
        depth = best_depth(mask_vol)
        if depth < 0 or depth >= volume.shape[2]:
            rows.append({"case": case, "status": "empty_mask"})
            continue

        raw_mask = mask_vol[:, :, depth]
        classes = map_classes(raw_mask)
        if classes.max() == 0:
            rows.append({"case": case, "status": "empty_classes"})
            continue

        image_slice = volume[:, :, depth]
        # Mask is at the raw in-plane resolution; resize to the image slice.
        target = (image_slice.shape[1], image_slice.shape[0])  # (W, H)
        class_img = Image.fromarray(classes).resize(
            target, Image.NEAREST)
        classes = np.asarray(class_img, dtype=np.uint8)

        fg = classes > 0
        inside = float(image_slice[fg].mean()) if fg.any() else 0.0
        outside = float(image_slice[~fg].mean()) if (~fg).any() else 0.0
        ratio = inside / max(outside, 1e-6)
        if ratio < args.min_intensity_ratio:
            rows.append({"case": case, "status": "weak_alignment",
                         "ratio": round(ratio, 3)})
            continue

        Image.fromarray(image_slice).convert("L").save(
            image_out / f"{case}.png")
        Image.fromarray(classes).save(mask_out / f"{case}.png")

        counts = {int(c): int((classes == c).sum())
                  for c in range(NUM_SEG_CLASSES)}
        rows.append({
            "case": case, "status": "ok", "depth": depth,
            "height": int(image_slice.shape[0]),
            "width": int(image_slice.shape[1]),
            "intensity_ratio": round(ratio, 3),
            **{f"px_class_{c}": n for c, n in counts.items()},
        })
        print(f"{case:10s} depth={depth:3d} ratio={ratio:4.2f} "
              f"classes={counts}")

    import pandas as pd
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    ok = manifest[manifest["status"] == "ok"]
    print(f"\nPairs written      : {len(ok)} / {len(cases)}")
    if len(ok):
        print("Mean intensity ratio: "
              f"{ok['intensity_ratio'].mean():.2f}")
        for c in range(1, NUM_SEG_CLASSES):
            print(f"Class {c} pixels    : "
                  f"{int(ok[f'px_class_{c}'].sum()):,}")
    skipped = manifest[manifest["status"] != "ok"]
    if len(skipped):
        print("Skipped:")
        print(skipped["status"].value_counts().to_string())
        print(skipped.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
