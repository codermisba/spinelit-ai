"""
yolo_prep.py
============

Build a YOLO detection dataset from SPIDER MRI volumes in YOLO-format
(darknet/Ultralytics) label files, with ZERO manual annotation cost:

the bounding boxes are derived **directly from the expert segmentation
masks** — for the mid-sagittal slice only (matching the severity input
view), every vertebra instance and every disc instance becomes an
axis-aligned box.

Classes
-------
    0 = vertebra     (SPIDER mask labels 1-25 and 101-125)
    1 = disc         (SPIDER mask labels 201-225)

Layout produced under --out_dir
-------------------------------
    images/   <case>_mid.jpg      (grayscale JPG, mirrored preprocessing)
    labels/   <case>.txt          (YOLO: cls xc yc w h, normalized 0-1)
    train.txt / val.txt           (image path lists, 80/20 split)

Method
------
1. Load image + mask volumes with SimpleITK.
2. Mid-sagittal depth = the slice with the most vertebra+disc mask signal.
3. Per structure label, take its pixels in that slice -> 2-D bounding box.
4. Convert px -> normalized YOLO (xc, yc, w, h); drop degenerate boxes
   smaller than 0.5% of the image area.

Usage
-----
    python yolo_prep.py --images_dir /content/SPIDER_data/images \
        --masks_dir  /content/SPIDER_data/masks \
        --out_dir    dataset/yolo
    python yolo_prep.py --self-test        # synthetic volume, no data needed
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# SPIDER mask label groups (bottom-up numbering, see prepare_spider.py).
VERTEBRA_LABELS = set(range(1, 26)) | set(range(101, 126))    # 1-25, 101-125
DISC_LABELS = set(range(201, 226))                            # 201-225

CLASS_OF = {**{lb: 0 for lb in VERTEBRA_LABELS},
            **{lb: 1 for lb in DISC_LABELS}}

MIN_BOX_AREA_FRACTION = 0.005


def _require_simpleitk():
    try:
        import SimpleITK  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required. Install it first:  pip install SimpleITK"
        ) from exc


def mid_sagittal_depth(mask_vol: np.ndarray) -> int:
    """Depth slice with the most vertebra+disc signal (mask in SPIDER labels)."""
    signal = np.isin(mask_vol, list(CLASS_OF)).sum(axis=(0, 1))
    if signal.max() <= 0:
        return mask_vol.shape[2] // 2
    return int(np.argmax(signal))


def boxes_from_labels(mask_slice: np.ndarray) -> list[tuple[int, float, float, float, float]]:
    """
    For every structure label in the slice: one (cls, xc, yc, w, h) box,
    normalized to [0,1]. Ignored if smaller than MIN_BOX_AREA_FRACTION.
    """
    h, w = mask_slice.shape
    area_frac = MIN_BOX_AREA_FRACTION * 1.0
    out: list[tuple[int, float, float, float, float]] = []
    for lb, cls in CLASS_OF.items():
        ys, xs = np.nonzero(mask_slice == lb)
        if ys.size == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if (bw * bh) / (h * w) < area_frac:
            continue
        xc = (x0 + x1) / 2.0 / w
        yc = (y0 + y1) / 2.0 / h
        out.append((cls, round(xc, 6), round(yc, 6),
                    round(bw / w, 6), round(bh / h, 6)))
    return sorted(out, key=lambda b: b[0])


def save_mid_sagittal(image_vol: np.ndarray, depth: int, dst: Path) -> None:
    image_slice = image_vol[:, :, depth].astype(np.float32)
    lo, hi = np.percentile(image_slice, [1, 99])
    arr8 = np.clip((image_slice - lo) / max(hi - lo, 1e-6), 0, 1)
    arr8 = (arr8 * 255).astype(np.uint8)
    Image.fromarray(arr8).convert("L").convert("RGB").save(str(dst))


def process_case(image_path: Path, mask_path: Path, image_out: Path,
                 label_out: Path, skips: dict) -> bool:
    import SimpleITK as sitk

    case = image_path.stem
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    except Exception as exc:  # noqa: BLE001
        skips[exc.__class__.__name__] = skips.get(exc.__class__.__name__, 0) + 1
        print(f"  [skip] {case}: {exc.__class__.__name__}: {str(exc)[:60]}")
        return False

    depth = mid_sagittal_depth(mask)
    save_mid_sagittal(img, depth, image_out / f"{case}_mid.jpg")

    boxes = boxes_from_labels(mask[:, :, depth])
    (label_out / f"{case}.txt").write_text(
        "\n".join(f"{c} {xc} {yc} {w} {h}" for c, xc, yc, w, h in boxes) + "\n"
    )
    return bool(boxes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir")
    ap.add_argument("--masks_dir")
    ap.add_argument("--out_dir", default="dataset/yolo")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test(Path(args.out_dir))
        return

    _require_simpleitk()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    image_out = out_dir / "images"
    label_out = out_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.mha"))
    if not image_files:
        raise SystemExit(f"No .mha volumes in {images_dir}")

    skips: dict = {}
    ok = 0
    splits = []
    for i, img_path in enumerate(image_files):
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            skips["missing_mask"] = skips.get("missing_mask", 0) + 1
            continue
        if process_case(img_path, mask_path, image_out, label_out, skips):
            ok += 1
            # 80/20 split file lists
            (image_out / f"{img_path.stem}_mid.jpg")
            splits.append(str(image_out / f"{img_path.stem}_mid.jpg"))

    rng = np.random.default_rng(42)
    rng.shuffle(splits)
    n_val = max(1, int(0.2 * len(splits)))
    (out_dir / "train.txt").write_text("\n".join(splits[n_val:]) + "\n")
    (out_dir / "val.txt").write_text("\n".join(splits[:n_val]) + "\n")

    print(f"Cases with boxes   : {ok} / {len(image_files)}")
    print(f"JPGs (YOLO images) : {len(list(image_out.glob('*.jpg')))}")
    print(f"Label files        : {len(list(label_out.glob('*.txt')))}")
    print(f"train.txt/val.txt  : {len(splits) - n_val} / {n_val}")
    if skips:
        print("Skips:", skips)
    if ok == 0:
        raise SystemExit("No cases produced boxes - check masks/labels.")


def _self_test(out_dir: Path) -> None:
    import SimpleITK as sitk

    tmp = Path(tempfile.mkdtemp(prefix="yolo_selftest_"))
    images, masks = tmp / "images", tmp / "masks"
    images.mkdir(parents=True), masks.mkdir(parents=True)

    vol = np.zeros((32, 32, 5), dtype=np.int16)
    mask = np.zeros_like(vol)
    # Two vertebra blobs and two disc blobs on the middle depth slice,
    # offset horizontally so the midsagittal depth is unambiguous.
    vol[:, :, 2] = 40 + np.random.default_rng(0).integers(0, 20, (32, 32))
    mask[6:10, 8:14, 2] = 5     # vertebra "L1"? (label 5 -> vertebra class)
    mask[16:20, 8:14, 2] = 3    # vertebra (label 3)
    mask[11:15, 8:14, 2] = 205  # disc L1/L2 (label 205 -> disc class)
    mask[21:25, 8:14, 2] = 201  # disc L5/S1
    mask[12:20, 12:18, 2] = 100  # spinal canal (ignored by YOLO classes)

    for name, arr in (("case001_t2.mha", vol), ("case001_t2.mha", mask)):
        pass
    sitk.WriteImage(sitk.GetImageFromArray(vol), str(images / "case001_t2.mha"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(masks / "case001_t2.mha"))

    from unittest import mock
    sys_argv_tmp = None  # reuse main() directly instead

    image_out = out_dir / "images"
    label_out = out_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    ok = process_case(images / "case001_t2.mha", masks / "case001_t2.mha",
                      image_out, label_out, {})
    label_txt = (label_out / "case001_t2.txt").read_text().strip()
    print("  self-test -> boxes:", ok)
    print("  label file contents:\n" + label_txt)

    n_boxes = len(label_txt.splitlines())
    assert ok and n_boxes == 4, f"expected 4 boxes, got {n_boxes}"
    for line in label_txt.splitlines():
        cls, xc, yc, w, h = map(float, line.split())
        assert cls in (0, 1)
        for v in (xc, yc, w, h):
            assert 0.0 < v <= 1.0, f"YOLO coords out of range: {line}"
    assert (out_dir / "images" / "case001_t2_mid.jpg").exists()
    print("  YOLO self-test PASSED")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()