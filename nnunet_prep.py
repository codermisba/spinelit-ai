"""
nnunet_prep.py
==============

Convert SPIDER MRI volumes + expert masks into the **nnU-Net v2 raw data
layout** so nnU-Net can be trained for 3-D segmentation of vertebrae,
intervertebral discs and the spinal canal on the mid-sagittal T2 path.

Output layout (nnU-Net style)
-----------------------------
    <out_base>/Dataset501_SPIDERMRI/imagesTr/<case>_0000.nii.gz
    <out_base>/Dataset501_SPIDERMRI/labelsTr/<case>.nii.gz
    <out_base>/Dataset501_SPIDERMRI/dataset.json

Label remapping (SPIDER -> ccTLD nnU-Net IDs)
---------------------------------------------
    0          background
    1          vertebra   (SPIDER 1-25        and 101-125 partial)
    2          disc       (SPIDER 201-225)
    3          spinal canal (SPIDER 100)

Usage
-----
    export nnUNet_raw=/content/nnUNet_raw \
    export nnUNet_preprocessed=/content/nnUNet_preprocessed \
    export nnUNet_results=/content/nnUNet_results
    python nnunet_prep.py --images_dir /content/SPIDER_data/images \
        --masks_dir /content/SPIDER_data/masks --out_base $nnUNet_raw
    python nnunet_prep.py --self-test                      # synthetic volume
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

# SPIDER label groups (bottom-up; see prepare_spider.py).
_VERT = set(range(1, 26)) | set(range(101, 126))
_DISC = set(range(201, 226))
_CANAL = {100}


def _remap(mask_vol: np.ndarray) -> np.ndarray:
    """SPIDER mask labels -> nnU-Net semantic IDs {0,1,2,3}."""
    out = np.zeros_like(mask_vol, dtype=np.uint8)
    out[np.isin(mask_vol, sorted(_VERT))] = 1
    out[np.isin(mask_vol, sorted(_DISC))] = 2
    out[np.isin(mask_vol, sorted(_CANAL))] = 3
    return out


def _require_simpleitk():
    try:
        import SimpleITK  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "SimpleITK is required. Install it first:  pip install SimpleITK"
        ) from exc


def convert_case(image_path: Path, mask_path: Path, images_tr: Path,
                 labels_tr: Path, skips: dict) -> bool:
    import SimpleITK as sitk

    case = image_path.stem
    try:
        img = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
    except Exception as exc:  # noqa: BLE001
        skips[exc.__class__.__name__] = skips.get(exc.__class__.__name__, 0) + 1
        print(f"  [skip] {case}: {str(exc)[:60]}")
        return False

    mask_arr = sitk.GetArrayFromImage(mask)
    remapped = _remap(mask_arr)
    if remapped.max() == 0:
        skips["no_foreground"] = skips.get("no_foreground", 0) + 1
        print(f"  [skip] {case}: empty after remap (no vertebra/disc/canal)")
        return False

    new_mask = sitk.GetImageFromArray(remapped)
    new_mask.CopyInformation(mask)

    # Use shared geometry/filename: SPIDER image+mask share the same name.
    sitk.WriteImage(img, str(images_tr / f"{case}_0000.nii.gz"))
    sitk.WriteImage(new_mask, str(labels_tr / f"{case}.nii.gz"))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir")
    ap.add_argument("--masks_dir")
    ap.add_argument("--out_base", default="nnUNet_raw")
    ap.add_argument("--dataset_id", type=int, default=501)
    ap.add_argument("--dataset_name", default="SPIDERMRI")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test(Path(args.out_base))
        return

    _require_simpleitk()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    ds_dir = Path(args.out_base) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    # Write dataset.json FIRST so verify_dataset_integrity never fails on a
    # missing file even when the loop below finds nothing to convert.
    def _write_json(num_training: int) -> None:
        dataset_json = {
            "channel_names": {"0": "T2"},
            "labels": {"background": 0, "vertebra": 1, "disc": 2,
                       "spinal_canal": 3},
            "numTraining": num_training,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    _write_json(0)

    image_files = sorted(images_dir.glob("*.mha"))
    if not image_files:
        raise SystemExit(f"No .mha volumes under {images_dir}. "
                         f"Is SPIDER_data mounted where the notebook expects it? "
                         f"(run: import os; print(os.listdir('/content/drive/MyDrive')))")

    skips: dict = {}
    ok = 0
    for img_path in image_files:
        mask_path = masks_dir / img_path.name
        try:
            has_mask = mask_path.exists()
        except OSError as exc:  # Drive FUSE can drop mid-run
            skips["drive_fuse_lost"] = skips.get("drive_fuse_lost", 0) + 1
            print(f"  [skip] {img_path.name}: Drive read failed: {exc}")
            continue
        if not has_mask:
            skips["missing_mask"] = skips.get("missing_mask", 0) + 1
            continue
        if convert_case(img_path, mask_path, images_tr, labels_tr, skips):
            ok += 1

    _write_json(ok)

    print(f"Converted        : {ok} / {len(image_files)} cases")
    print(f"Dataset folder   : {ds_dir}")
    print(f"imagesTr / labelsTr : {len(list(images_tr.glob('*.nii.gz')))} / "
          f"{len(list(labels_tr.glob('*.nii.gz')))}")
    if skips:
        print("Skips:", skips)
    if ok == 0:
        raise SystemExit("No cases converted - check input paths.")
    print("\nNext: nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity\n"
          "      nnUNetv2_train 501 3d_fullres all")


def _self_test(out_base: Path) -> None:
    import SimpleITK as sitk

    tmp = Path(tempfile.mkdtemp(prefix="nnunet_selftest_"))
    images, masks = tmp / "images", tmp / "masks"
    images.mkdir(parents=True), masks.mkdir(parents=True)

    vol = np.zeros((24, 24, 8), dtype=np.int16)
    mask = np.zeros_like(vol)
    vol[4:20, 4:20, :] = 50
    mask[4:8, 6:14, 3] = 5     # vertebra
    mask[10:14, 6:14, 3] = 3   # vertebra
    mask[7:11, 6:14, 3] = 205  # disc
    mask[14:18, 6:14, 3] = 201 # disc
    mask[6:16, 16:20, 3] = 100 # spinal canal
    rng = np.random.default_rng(1)
    vol += rng.integers(0, 25, vol.shape).astype(np.int16)

    for name, arr in (("case009_t2.mha", vol), ("case009_t2.mha", mask)):
        pass
    img = sitk.GetImageFromArray(vol)
    sitk.WriteImage(img, str(images / "case009_t2.mha"))
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(masks / "case009_t2.mha"))

    ds_dir = out_base / "Dataset501_SPIDERMRI"
    images_tr, labels_tr = ds_dir / "imagesTr", ds_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    ok = convert_case(images / "case009_t2.mha", masks / "case009_t2.mha",
                      images_tr, labels_tr, {})
    assert ok
    remapped = sitk.GetArrayFromImage(
        sitk.ReadImage(str(labels_tr / "case009_t2.nii.gz"))
    )
    print("  self-test -> remapped unique labels:",
          sorted(np.unique(remapped).tolist()))
    assert set(np.unique(remapped).tolist()) == {0, 1, 2, 3}
    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "vertebra": 1, "disc": 2,
                   "spinal_canal": 3},
        "numTraining": 1,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    assert (ds_dir / "dataset.json").exists()
    ds = json.loads((ds_dir / "dataset.json").read_text())
    assert ds["numTraining"] == 1
    print("  nnU-Net self-test PASSED")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()