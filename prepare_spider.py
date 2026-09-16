"""
prepare_spider.py
=================

Prepare the **SPIDER** dataset (Zenodo record 10159290, CC-BY 4.0) for the
DDD-only spine pipeline (per-level Pfirrmann grading).

SPIDER (van der Graaf et al., Sci Data 2024) provides, per study: sagittal
MRI series (T1 / T2 / T2-SPACE) stored as `.mha` volumes, segmentation
masks of the vertebrae / intervertebral discs (IVDs) / spinal canal, and
expert radiological gradings (Pfirrmann 1-5, disc narrowing, herniation,
Modic, ...) for each IVD level.

Mask label legend (SPIDER):
    0     = background
    1-25  = vertebrae, numbered from bottom (1 = L5 .. 5 = L1)
    100   = spinal canal
    101-125 = partially visible vertebrae
    201-225 = intervertebral discs, numbered from top (201 = most superior)

What this script produces (the exact inputs SpineDataset expects):
  dataset/data/processed_spider_jpgs/<id>_mid.jpg   midsagittal T2/T2-SPACE JPG
  dataset/coords_pretrain.csv   -> filename, level, relative_x, relative_y
  dataset/ddd_labels.csv        -> filename, level, pfirrmann_grade

Coordinate convention
---------------------
Disc centroids are read directly from the segmentation mask: for each lumbar
IVD label we select the sagittal depth slice where that disc has the largest
cross-section (its mid-disc plane) and take the 2D centroid in that plane.
Coordinates are normalized to 0-1 (relative_x / relative_y, origin
top-left), matching the dataset loader.

Level numbering
---------------
SPIDER numbers discs *top-down* (verified empirically on real data):
    the highest disc label in a patient's mask = L5/S1, the next lower
    label = L4/L5, etc.  Patients show 5 lumbar discs (labels 201-205)
    or 6 when T12/L1 is included (labels 201-206 -> 202-206 are the
    lumbar L1/L2..L5/S1, and 201 = T12/L1 is dropped).
The radiology "IVD label" column uses the same convention: 1-based,
top-down, so the highest IVD label for the patient = L5/S1.
Mappings are therefore built dynamically per patient (see
_build_disc_label_to_level / _build_ivd_label_to_level) and emit the
pipeline's top-down order [L1/L2, L2/L3, L3/L4, L4/L5, L5/S1].

Radiology CSV columns: Patient, IVD label, Modic, UP endplate, LOW
endplate, Spondylolisthesis, Disc herniation, Disc narrowing, Disc bulging,
Pfirrman grade.

Usage (on Colab)
----------------
  python prepare_spider.py            # downloads images/masks/CSVs to data_dir
  python prepare_spider.py --data_dir /content/SPIDER_data --skip_download

Dependencies (install on Colab first):
  pip install SimpleITK pillow numpy pandas requests
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import DATA_DIR, DISC_LEVELS, ROOT_DIR

# -- SPIDER Zenodo record ------------------------------------------------
RECORD = "https://zenodo.org/records/10159290/files"
FILES = {
    "images": f"{RECORD}/images.zip",
    "masks": f"{RECORD}/masks.zip",
    "overview": f"{RECORD}/overview.csv",
    "gradings": f"{RECORD}/radiological_gradings.csv",
}

# Which scan types to train on (sagittal T2-weighted and T2-SPACE).
SCAN_TYPES = ("t2", "t2_SPACE")

# Mask disc labels are TOP-DOWN (201 = most superior disc, highest label
# number = most inferior = L5/S1).  The number of labels varies per patient
# (5 for lumbar-only, 6 when T12/L1 is included).
# We build the mapping dynamically per patient rather than hard-coding it.
_DISC_LABEL_OFFSET = 200   # mask disc labels start at 201 = offset + 1


def _build_disc_label_to_level(mask_vol: np.ndarray) -> dict[int, str]:
    """
    Given a mask volume, find all disc labels (201-225), determine which
    is L5/S1 (the highest label number = most inferior), and return a
    mapping {label_int: level_name} for the 5 lumbar disc levels.
    Labels above L1/L2 (e.g. T12/L1) are excluded from the mapping.
    """
    present = sorted(int(lb) for lb in np.unique(mask_vol)
                     if _DISC_LABEL_OFFSET < lb < _DISC_LABEL_OFFSET + 26)
    if not present:
        return {}
    max_label = present[-1]          # highest label = L5/S1
    mapping = {}
    for label in present:
        idx = max_label - label      # 0 = L5/S1 (last in DISC_LEVELS)
        if 0 <= idx < len(DISC_LEVELS):
            mapping[label] = DISC_LEVELS[-(idx + 1)]   # top-down ordering
    return mapping


def _build_ivd_label_to_level(gradings_df: pd.DataFrame,
                              patient_id: str) -> dict[int, str]:
    """
    For a patient's radiological gradings, find all IVD labels (1-based),
    determine which is L5/S1 (highest IVD number), and return a mapping
    {ivd_label_int: level_name} for the 5 lumbar disc levels.
    """
    pat = gradings_df[gradings_df["Patient"].astype(str) == str(patient_id)]
    if pat.empty:
        return {}
    ivd_labels = sorted(pat["IVD label"].dropna().astype(int).unique())
    if not ivd_labels:
        return {}
    max_ivd = ivd_labels[-1]        # highest IVD label = L5/S1
    mapping = {}
    for lb in ivd_labels:
        idx = max_ivd - lb           # 0 = L5/S1 (last in DISC_LEVELS)
        if 0 <= idx < len(DISC_LEVELS):
            mapping[lb] = DISC_LEVELS[-(idx + 1)]   # top-down ordering
    return mapping

PFRRMANN_COL = "Pfirrman grade"   # exact column name in radiological_gradings.csv


def _download(data_dir: Path) -> None:
    import requests
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = data_dir / Path(url).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [have] {dest.name}")
            continue
        print(f"  [download] {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    # Extract zips into <data_dir>/extracted
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    for zp, target in ((data_dir / "images.zip", images_dir),
                       (data_dir / "masks.zip", masks_dir)):
        if target.exists() and any(target.iterdir()):
            continue
        print(f"  [extract] {zp.name}")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(target)
        # Normalize so the actual files live directly under images/ and masks/
        for sub in [p for p in target.iterdir() if p.is_dir()]:
            for f in sub.iterdir():
                if f.is_file():
                    f.replace(target / f.name)
            sub.rmdir()
        if (target / "images").is_dir():
            for f in (target / "images").iterdir():
                f.replace(target / f.name)
            (target / "images").rmdir()
        if (target / "masks").is_dir():
            for f in (target / "masks").iterdir():
                f.replace(target / f.name)
            (target / "masks").rmdir()


def _require_simpleitk() -> None:
    try:
        import SimpleITK  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required to read the SPIDER .mha volumes.\n"
            "Install it first:  pip install SimpleITK\n"
            "(checkout/colab_training.ipynb cell 4 does this automatically.)"
        ) from exc


def _load_volume(path: Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def _disc_centroid_on_best_slice(image_vol, mask_vol, disc_label):
    """
    For a given disc label, find the sagittal depth slice with the largest
    masked area and return (centroid_col, centroid_row, slice) in pixel coords.
    """
    mask_vol = (mask_vol == disc_label)
    area_per_depth = mask_vol.sum(axis=(0, 1))
    if area_per_depth.max() == 0:
        return None
    depth = int(np.argmax(area_per_depth))
    mask_slice = mask_vol[:, :, depth]
    ys, xs = np.nonzero(mask_slice)
    h, w = mask_slice.shape
    centroid = (float(xs.mean()) / w, float(ys.mean()) / h)  # (rel_x, rel_y)
    return centroid


def _process_case(
    image_path: Path,
    mask_path: Path,
    gradings_df: pd.DataFrame,
    patient_id: str,
    jpg_out: Path,
    coords_rows: list,
    grade_rows: list,
    skips: dict,
) -> None:
    try:
        image_vol = _load_volume(image_path)
        mask_vol = _load_volume(mask_path)
    except Exception as exc:  # noqa: BLE001
        reason = f"{exc.__class__.__name__}: {str(exc)[:80]}"
        skips[reason] = skips.get(reason, 0) + 1
        print(f"  [skip] {image_path.name}: {exc}")
        return

    try:
        # Save the mid-sagittal T2 slice as a JPG (the depth with the most
        # vertebra+disc signal is a good proxy for the mid-sagittal plane).
        fname = f"{patient_id}_mid.jpg"
        combined_signal = (mask_vol > 0).sum(axis=(0, 1))
        depth = int(np.argmax(combined_signal)) if combined_signal.max() > 0 else (
            image_vol.shape[2] // 2
        )
        h, w = image_vol.shape[:2]
        image_slice = image_vol[:, :, depth].astype(np.float32)
        lo, hi = np.percentile(image_slice, [1, 99]) if image_slice.size else (0, 1)
        arr8 = np.clip((image_slice - lo) / max(hi - lo, 1e-6), 0, 1)
        arr8 = (arr8 * 255).astype(np.uint8)
        Image.fromarray(arr8).convert("L").convert("RGB").save(jpg_out / fname)

        # Disc centroids from the mask — dynamic mapping per patient.
        disc_map = _build_disc_label_to_level(mask_vol)
        centroids = {}
        for lab, level in disc_map.items():
            c = _disc_centroid_on_best_slice(image_vol, mask_vol, lab)
            if c is not None:
                centroids[level] = c

        for level in DISC_LEVELS:
            if level in centroids:
                rx, ry = centroids[level]
                coords_rows.append(
                    {"filename": fname, "level": level,
                     "relative_x": round(rx, 6), "relative_y": round(ry, 6)}
                )

        # Pfirrmann grades from the radiology CSV — dynamic mapping per patient.
        pat = gradings_df[gradings_df["Patient"].astype(str) == str(patient_id)]
        if not pat.empty:
            ivd_map = _build_ivd_label_to_level(gradings_df, patient_id)
            for label, level in ivd_map.items():
                row = pat[pat["IVD label"].astype(str) == str(label)]
                if row.empty:
                    continue
                val = str(row.iloc[0][PFRRMANN_COL]).strip()
                if val and val.lower() not in ("nan", "none", ""):
                    try:
                        g = int(float(val))
                    except ValueError:
                        continue
                    if 1 <= g <= 5:
                        grade_rows.append(
                            {"filename": fname, "level": level,
                             "pfirrmann_grade": g}
                        )
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the run
        reason = f"{exc.__class__.__name__}: {str(exc)[:80]}"
        skips[reason] = skips.get(reason, 0) + 1
        print(f"  [skip] {image_path.name} (post-load): {exc}")
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/content/SPIDER_data",
                    help="dir to hold zips + extracted images/masks (default "
                         "/content/SPIDER_data)")
    ap.add_argument("--skip_download", action="store_true",
                    help="data already downloaded/extracted in --data_dir")
    args = ap.parse_args()

    # Never let a stale CSV from a previous (possibly crashed) run masquerade
    # as this run's output: remove them up front; they are fully regenerated.
    for stale in (ROOT_DIR / "dataset" / "coords_pretrain.csv",
                  ROOT_DIR / "dataset" / "ddd_labels.csv"):
        stale.unlink(missing_ok=True)

    data_dir = Path(args.data_dir)
    _require_simpleitk()
    if not args.skip_download:
        _download(data_dir)

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    gradings_csv = data_dir / "radiological_gradings.csv"
    if not gradings_csv.exists():
        raise SystemExit(
            f"radiological_gradings.csv not found under {data_dir}. "
            "This file provides the Pfirrmann ground-truth labels."
        )

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise SystemExit(
            f"images/ and masks/ folders not found under {data_dir}. "
            "Re-run without --skip_download to fetch+extract from Zenodo."
        )

    gradings = pd.read_csv(gradings_csv) if gradings_csv.exists() else pd.DataFrame()
    print(f"Radiology records : {len(gradings)}")

    # Group the .mha files by (patient, scan_type) and pick a T2/T2-SPACE one.
    images = sorted(images_dir.glob("*.mha"))
    masks = {p.name: p for p in masks_dir.glob("*.mha")}
    print(f"Image volumes     : {len(images)}")

    jpg_out = DATA_DIR / "processed_spider_jpgs"
    jpg_out.mkdir(parents=True, exist_ok=True)

    coords_rows: list = []
    grade_rows: list = []
    skips: dict = {}

    used = 0
    for img in images:
        name = img.stem                    # e.g. 305_t2_SPACE
        scan_type = "_".join(name.split("_")[1:])
        if scan_type not in SCAN_TYPES:
            continue
        patient_id = name.split("_")[0]
        if img.name not in masks:
            continue
        _process_case(img, masks[img.name], gradings, patient_id,
                      jpg_out, coords_rows, grade_rows, skips)
        used += 1

    coords_df = pd.DataFrame(coords_rows)
    grade_df = pd.DataFrame(grade_rows)
    coords_df.to_csv(ROOT_DIR / "dataset" / "coords_pretrain.csv", index=False)
    grade_df.to_csv(ROOT_DIR / "dataset" / "ddd_labels.csv", index=False)

    print(f"T2/T2-SPACE series used : {used}")
    print(f"Disc landmark rows      : {len(coords_df)}")
    print(f"Pfirrmann label rows    : {len(grade_df)}")
    print(f"Images written to       : {jpg_out}")

    jpg_count = sum(1 for _ in jpg_out.glob("*.jpg"))
    print(f"JPGs on disk            : {jpg_count}")
    files_with_centroids = len(set(r["filename"] for r in coords_rows))
    print(f"Cases with disc labels  : {files_with_centroids} of {used}")
    if skips:
        print("Skip reasons (top 10):")
        for reason, n in sorted(skips.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {n:5d} x {reason}")

    if not used:
        raise SystemExit(
            "No T2/T2-SPACE cases were processed. Check that images/ contains "
            "*_t2.mha / *_t2_SPACE.mha and that matching *_t2*.mha masks exist "
            "in masks/."
        )
    if len(coords_df) == 0:
        raise SystemExit(
            "prepare_spider ran but produced 0 landmark rows. Check that the "
            "segmentation masks in masks/ actually contain the lumbar disc "
            "labels 201..225."
        )


if __name__ == "__main__":
    main()
