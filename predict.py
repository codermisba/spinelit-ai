"""
predict.py
==========

Run the Spine Foundation Model on a single X-ray / MRI image.

Example
-------
python predict.py --image "dataset/data/processed_tseg_jpgs/case_0000.jpg"

With longitudinal clinical inputs:
python predict.py --image scan.jpg --age 58 --sex female --pain-scale 6 \
    --modality mri --pain-years 4 --start-year 2022 --years-ahead 5

Prints every lumbar vertebra (L1-L5) and disc level (L1/L2-L5/S1) with
pixel coordinates, confidence, geometric indicators, DDD grading,
spondylolisthesis estimate and — when clinical inputs are given — the
longitudinal progression risk. Saves an annotated image into outputs/.
The original image is never modified.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from config import (
    DISC_LEVELS,
    LONGITUDINAL_MODEL_NAME,
    OUTPUT_DIR,
    VERTEBRAE,
)
from clinical_model import ClinicalInputs, LongitudinalRiskModel, predict_risk
from utils import (
    MEDICAL_DISCLAIMER,
    compute_geometric_indicators,
    decode_outputs,
    draw_landmarks,
    ddd_labels_available,
    load_model,
    meyerding_grade,
    norm_to_pixels,
    predict,
    preprocess_image,
    severity_from_grade,
    spondy_labels_available,
)


def load_ground_truth(image_path: Path):
    """Ground-truth landmarks for this image if present in the CSVs."""
    csv_path = Path("dataset/coords_pretrain.csv")

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip()

    rows = df[df["filename"] == image_path.name]
    if rows.empty:
        return None

    coords = []
    for level in DISC_LEVELS:
        row = rows[rows["level"] == level]
        if row.empty:
            coords.extend([0.0, 0.0])
        else:
            coords.extend(
                [float(row.iloc[0]["relative_x"]),
                 float(row.iloc[0]["relative_y"])]
            )
    return coords


def load_longitudinal_model():
    path = Path(LONGITUDINAL_MODEL_NAME)
    if not path.exists():
        return None
    model = LongitudinalRiskModel()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Longitudinal model loaded : {path}"
          f" (image-fused={checkpoint.get('image_fused', False)})")
    return model


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Predict lumbar landmarks + degeneration on one image."
    )
    parser.add_argument("--image", required=True, help="X-ray/MRI image path.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default=None)

    # Longitudinal / clinical inputs
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", type=str, default=None,
                        choices=["male", "female"])
    parser.add_argument("--pain-scale", type=float, default=None,
                        help="VAS pain scale 0-10.")
    parser.add_argument("--modality", type=str, default=None,
                        choices=["xray", "mri"],
                        help="Imaging done: xray or MRI.")
    parser.add_argument("--pain-years", type=float, default=None,
                        help="Years the patient has had pain.")
    parser.add_argument("--start-year", type=int, default=None,
                        help="Calendar year symptoms started.")
    parser.add_argument("--years-ahead", type=float, default=5.0,
                        help="Longitudinal prediction horizon (years).")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # ---------------------------------------------------------
    # Imaging model
    # ---------------------------------------------------------
    try:
        model, device, checkpoint_path = load_model(args.checkpoint)
    except FileNotFoundError as exc:
        print(exc)
        return

    print(f"Device      : {device}")
    print(f"Checkpoint  : {checkpoint_path.name}")

    original = Image.open(image_path).convert("RGB")
    outputs = predict(model, preprocess_image(original), device)
    decoded = decode_outputs(outputs)

    width, height = original.size
    pixels = norm_to_pixels(decoded["points"], width, height)

    vertebra_pts = pixels[:len(VERTEBRAE)]
    disc_pts = pixels[len(VERTEBRAE):]
    v_conf = decoded["localization_conf"][:len(VERTEBRAE)]
    d_conf = decoded["localization_conf"][len(VERTEBRAE):]

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print()
    print(f"Image       : {image_path.name}")
    print(f"Resolution  : {width} x {height}")
    print()

    print("== Lumbar Vertebrae ==")
    for i, name in enumerate(VERTEBRAE):
        print(f"{name:<5} centre  ({vertebra_pts[i,0]:6.1f},"
              f"{vertebra_pts[i,1]:6.1f})  confidence {v_conf[i]:.3f}")
    print()

    print("== Disc Levels ==")
    for i, level in enumerate(DISC_LEVELS):
        print(f"{level:<6} centre ({disc_pts[i,0]:6.1f},{disc_pts[i,1]:6.1f})"
              f"  confidence {d_conf[i]:.3f}")
    print()

    print("== Geometric Indicators (label-free) ==")
    for ind in compute_geometric_indicators(pixels):
        flags = []
        if ind["narrowed"]:
            flags.append("narrowed disc space")
        if ind["offset_flag"]:
            flags.append("centre offset (possible listhesis)")
        flag_text = f"  <- {'; '.join(flags)}" if flags else ""
        print(f"{ind['level']:<6} space x{ind['relative_space']:.2f} "
              f"offset {ind['offset_ratio']:.2f}{flag_text}")
    print()

    print("== Disc Degeneration Estimate (DDD) ==")
    if not ddd_labels_available():
        print("(heads untrained - provide dataset/ddd_labels.csv and retrain"
              " for calibrated grades)")
    for i, level in enumerate(DISC_LEVELS):
        grade = decoded["ddd_grade"][i]
        print(f"{level:<6} grade {grade:4.2f}/4 "
              f"({severity_from_grade(grade):<8}) confidence "
              f"{decoded['ddd_conf'][i]:.3f}")
    print()

    print("== Spondylolisthesis Estimate ==")
    if not spondy_labels_available():
        print("(heads untrained - provide dataset/spondy_labels.csv and"
              " retrain for calibrated slip estimates)")
    for i, level in enumerate(DISC_LEVELS):
        slip = decoded["spondy_slip_pct"][i]
        print(f"{level:<6} slip {slip:5.1f}% ({meyerding_grade(slip)})"
              f" confidence {decoded['spondy_conf'][i]:.3f}")
    print()

    # ---------------------------------------------------------
    # Longitudinal prediction (needs clinical inputs + trained model)
    # ---------------------------------------------------------
    clinical_given = all(
        v is not None
        for v in [args.age, args.sex, args.pain_scale, args.modality,
                  args.pain_years, args.start_year]
    )

    if clinical_given:
        risk_model = load_longitudinal_model()
        if risk_model is not None:
            results = predict_risk(
                risk_model,
                ClinicalInputs(
                    age=args.age, sex=args.sex,
                    pain_scale=args.pain_scale, modality=args.modality,
                    pain_years=args.pain_years, start_year=args.start_year,
                ),
                image_features=outputs["features"][0].float().cpu(),
                years_ahead=args.years_ahead,
            )
            print(f"== Longitudinal Risk (+{args.years_ahead:g} years) ==")
            print(f"Overall risk : {results['overall_risk']*100:.1f}%")
            for i, level in enumerate(DISC_LEVELS):
                print(
                    f"{level:<6} predicted grade "
                    f"{results['future_grade'][i]:4.2f}/4"
                    f" | worsening probability "
                    f"{results['progression_risk'][i]*100:5.1f}%"
                )
            print()
        else:
            print("Clinical inputs given but no longitudinal model found.")
            print(f"Train it with: python train_clinical.py")
            print()
    else:
        print("(Add --age --sex --pain-scale --modality --pain-years"
              " --start-year for the longitudinal risk prediction.)")
        print()

    # ---------------------------------------------------------
    # Annotated output
    # ---------------------------------------------------------
    ground_truth = load_ground_truth(image_path)

    annotated = draw_landmarks(
        original,
        coords=decoded["points"],
        confidence=decoded["localization_conf"],
        ground_truth=ground_truth,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{image_path.stem}_{stamp}_annotated.jpg"

    annotated.save(output_path)

    print(f"Annotated image saved to : {output_path}")
    print()
    print(MEDICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
