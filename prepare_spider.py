"""
prepare_spider.py
=================

Prepare the **SPIDER** dataset (Zenodo record 10159290, CC-BY 4.0) for the
DDD-only spine pipeline.

SPIDER (CD Oswald et al.) provides, per case: MRI series + segmentation
masks + radiological gradings (Pfirrmann 1-5, disc narrowing, herniation,
Modic, ...) for the lumbar IVD levels.

What this script produces (the exact inputs SpineDataset expects):
  dataset/data/processed_spider_jpgs/<filename>.jpg   midsagittal T2/JPGs
  dataset/coords_pretrain.csv    -> filename, level, relative_x, relative_y
  dataset/ddd_labels.csv         -> filename, level, pfirrmann_grade

Coordinate convention
---------------------
Disc centroids are derived from the vertebral-body segmentation masks:
  * Each vertebral body label -> its 2D centroid.
  * Disc centroid (between two vertebrae) -> midpoint of the adjacent
    vertebral centroids.
SPIDER indexes the lumbar levels **bottom-up**: lumbar disc 1 == L5/S1,
disc 5 == L1/L2 (and vertebra 1 == L5 .. vertebra 5 == L1). We re-map to
the pipeline's top-down order [L1/L2, L2/L3, L3/L4, L4/L5, L5/S1].

Relative (0-1) coordinates are measured in the 2D image plane with origin
at the top-left and normalized by the slice width/height, matching the
`relative_x` / `relative_y` columns the dataset loader reads.

NOTE: SPIDER masks are 3D (z,y,x) with anisotropic voxels; this script
selects the **mid-sagittal axial-slice index in the T2 series** and sums
masks across the slice band centred on it so the mid-disc plane is used.

Usage (on Colab, where the dataset is downloaded):
  python prepare_spider.py --data_dir /content/SPIDER_data
  # or:  python prepare_spider.py --data_dir /content/SPIDER_data --skip_download

Dependencies (install on Colab first):
  pip install nibabel zipfile36 requests pillow numpy pandas scipy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    DATA_DIR,
    DISC_LEVELS,
    ROOT_DIR,
)

# Pipeline top-down order -> SPIDER bottom-up lumbar disc index (1..5)
# DISC_LEVELS[0]=L1/L2 -> spider disc 5, ... , DISC_LEVELS[4]=L5/S1 -> 1
_PIPE_LEVEL_TO_SPIDER_DISC = {level: 5 - i for i, level in enumerate(DISC_LEVELS)}

# SPIDER vertebra body index (bottom-up: 1=L5 .. 5=L1).
# Used only to sanity-check the derived disc midpoints.
PFRRMANN_MIN = 1
PFRRMANN_MAX = 5


def _download_if_needed(data_dir: Path) -> None:
    """Lightweight downloader; safer to pre-download SPIDER manually on Colab."""
    import requests
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        if any(data_dir.iterdir()):
            return
    except Exception:  # noqa: BLE001
        pass
    print(
        "[warning] Automatic download from Zenodo is not fully scripted "
        "here (3.8 GB, multi-part). Download SPIDER manually and pass "
        "--data_dir pointing at the extracted folder, e.g. from Kaggle/HF: "
        "https://huggingface.co/datasets/cdoswald/SPIDER"
    )


def _read_radiology_labels(rad_csv: Path) -> dict:
    """Return {spider_disc_index: pfirrmann_grade} from the case radiology."""
    df = pd.read_csv(rad_csv)
    # Column names vary; map common candidates.
    level_col = next((c for c in df.columns if "level" in c.lower()
                      or "ivd" in c.lower() or "disc" in c.lower()), df.columns[0])
    pf_col = next((c for c in df.columns if "pfirrmann" in c.lower()
                   or "grade" in c.lower() or "ddd" in c.lower()), None)
    if pf_col is None:
        return {}
    grades = {}
    for _, row in df.iterrows():
        key = row[level_col]
        try:
            g = int(float(row[pf_col]))
        except (TypeError, ValueError):
            continue
        if not (PFRRMANN_MIN <= g <= PFRRMANN_MAX):
            continue
        grades[int(key)] = g
    return grades


def _load_nifti(path: Path):
    import nibabel as nib
    return nib.load(str(path))


def _midsagittal_slice(image_arr, seg_arr):
    """Pick the sagittal plane (axis=2) that lies mid-disc for T2."""
    # image_arr/seg_arr: (z, y, x). Sagittal slices are along axis=2 (x).
    # Heuristic: pick the sagittal plane with the most segmented
    # vertebral-body voxels (mid-sagittal).
    sag_density = seg_arr.sum(axis=(0, 1))
    if sag_density.sum() == 0:
        x_idx = image_arr.shape[2] // 2
    else:
        x_idx = int(np.argmax(sag_density))
    return image_arr[:, :, x_idx], seg_arr[:, :, x_idx], x_idx


def _vertebra_centroids_2d(seg_slice, vert_labels):
    """Map each vertebra label -> 2D centroid (row, col) in the sagittal slice."""
    centroids = {}
    for lab in vert_labels:
        mask = seg_slice == lab
        if mask.sum() == 0:
            continue
        ys, xs = np.nonzero(mask)
        centroids[lab] = (float(ys.mean()), float(xs.mean()))
    return centroids


def _spider_label_to_pipe_level(vert_centroids: dict) -> dict:
    """
    SPIDER vertebra labels are bottom-up (1=L5 .. 5=L1). Re-map to the
    pipeline's vertebra names so we can derive disc midpoints in a uniform
    (top-down) order.
    """
    names = {1: "L5", 2: "L4", 3: "L3", 4: "L2", 5: "L1"}
    out = {}
    for lab, c in vert_centroids.items():
        if lab in names:
            out[names[lab]] = c
    return out


def _derive_disc_centroids(vert: dict) -> dict:
    """Midpoint between adjacent vertebral centroids -> disc level -> (x,y)."""
    order = DISC_LEVELS
    vert_pos = {v: np.array(vert[v], dtype=float) for v in
                ["L1", "L2", "L3", "L4", "L5"] if v in vert}
    points = {}
    for i in range(len(order)):
        # map disc level -> its two bounding vertebra names in pipeline order
        if order[i] == "L1/L2":
            a, b = "L1", "L2"
        elif order[i] == "L2/L3":
            a, b = "L2", "L3"
        elif order[i] == "L3/L4":
            a, b = "L3", "L4"
        elif order[i] == "L4/L5":
            a, b = "L4", "L5"
        else:  # L5/S1
            a, b = "L5", "S1"
        pa, pb = vert_pos.get(a), vert_pos.get(b)
        if pa is not None and pb is not None:
            points[order[i]] = ((pa + pb) / 2.0)
        elif pa is not None and a == "L5" and "S1" in vert:
            points[order[i]] = ((pa + np.array(vert["S1"])) / 2.0)
    return points


def _to_relative(points: dict, h: int, w: int) -> dict:
    """Convert (row, col) pixel coords to (relative_x, relative_y) in 0-1."""
    out = {}
    for level, (r, c) in points.items():
        out[level] = (float(c) / float(w), float(r) / float(h))
    return out


def _process_case(
    case_dir: Path,
    jpg_out: Path,
    coords_rows: list,
    grade_rows: list,
) -> None:
    # Locate the T2 image, segmentation, and radiology annotations.
    nii = sorted(case_dir.rglob("*T2*.nii*"))
    seg = sorted(case_dir.rglob("*seg*.nii*") + case_dir.rglob("*mask*.nii*"))
    rad = list(case_dir.rglob("*.csv"))
    if not nii or not seg:
        return
    # pick radiology CSV (skip anything that is a coords dump)
    rad_csv = None
    for c in rad:
        try:
            with open(c) as fh:
                head = fh.read(2000).lower()
        except Exception:
            continue
        if "pfirrmann" in head or "grade" in head or "level" in head:
            rad_csv = c
            break

    try:
        img_ni = _load_nifti(nii[0])
        seg_ni = _load_nifti(seg[0])
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {case_dir.name}: {exc}")
        return

    image_arr = np.asanyarray(img_ni.get_fdata()).astype(np.float32)
    seg_arr = np.asanyarray(seg_ni.get_fdata()).astype(np.int32)

    # Mid-sagittal slice in the T2 volume (same plane for image and mask).
    image_slice, seg_slice, _ = _midsagittal_slice(image_arr, seg_arr)

    h, w = image_slice.shape
    if w == 0 or h == 0:
        return

    vert_labels = [1, 2, 3, 4, 5]   # bottom-up lumbar bodies (verify!)
    vert = _spider_label_to_pipe_level(
        _vertebra_centroids_2d(seg_slice, vert_labels)
    )
    disc_points_px = _derive_disc_centroids(vert)
    rel = _to_relative(disc_points_px, h, w)
    # Normalize to 8-bit and persist the JPG.
    lo, hi = np.percentile(image_slice, [1, 99])
    arr8 = np.clip((image_slice - lo) / max(hi - lo, 1e-6), 0, 1)
    arr8 = (arr8 * 255).astype(np.uint8)
    fname = f"{case_dir.name}_midsag.jpg"
    Image.fromarray(arr8, mode="L").convert("RGB").save(jpg_out / fname)

    # Landmark + grade rows (only for levels we have a centroid for).
    for level in DISC_LEVELS:
        if level in rel:
            x, y = rel[level]
            coords_rows.append(
                {"filename": fname, "level": level,
                 "relative_x": round(x, 6), "relative_y": round(y, 6)}
            )

    if rad_csv is not None:
        grades = _read_radiology_labels(rad_csv)
        if grades:
            for level in DISC_LEVELS:
                spider_disc = _PIPE_LEVEL_TO_SPIDER_DISC[level]
                g = grades.get(spider_disc)
                if g is not None:
                    grade_rows.append(
                        {"filename": fname, "level": level,
                         "pfirrmann_grade": int(g)}
                    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="path to the extracted SPIDER dataset folder")
    ap.add_argument("--skip_download", action="store_true",
                    help="already downloaded; do not attempt download")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not args.skip_download:
        _download_if_needed(data_dir)

    jpg_out = DATA_DIR / "processed_spider_jpgs"
    jpg_out.mkdir(parents=True, exist_ok=True)

    coords_rows: list = []
    grade_rows: list = []

    # SPIDER : <data_dir>/<case>/..., but may also be <data_dir>/<study>/<case>
    case_dirs = [p for p in data_dir.glob("*") if p.is_dir()]
    nested = [c for c in case_dirs if any(x.is_dir() for x in c.iterdir())]
    if len(nested) == len(case_dirs) and case_dirs:
        case_dirs = [c for d in case_dirs for c in d.iterdir() if c.is_dir()]

    n_ok = 0
    for idx, case_dir in enumerate(case_dirs):
        _process_case(case_dir, jpg_out, coords_rows, grade_rows)
        n_ok += 1
        if (idx + 1) % 20 == 0:
            print(f"  ... {idx + 1}/{len(case_dirs)} cases processed")

    coords_df = pd.DataFrame(coords_rows)
    grade_df = pd.DataFrame(grade_rows)

    coords_df.to_csv(ROOT_DIR / "dataset" / "coords_pretrain.csv",
                     index=False)
    grade_df.to_csv(ROOT_DIR / "dataset" / "ddd_labels.csv", index=False)

    print(f"Cases scanned          : {n_ok}")
    print(f"Disc landmark rows     : {len(coords_df)}")
    print(f"Pfirrmann label rows   : {len(grade_df)}")
    print(f"Images written to      : {jpg_out}")
    print(f"coords_pretrain.csv    : {len(coords_df)} rows")
    print(f"ddd_labels.csv         : {len(grade_df)} rows")


if __name__ == "__main__":
    main()
