"""
predict.py
==========

Run the Spine Foundation Model on a single MRI image.

Example
-------
python predict.py --image "dataset/data/processed_tseg_jpgs/case_0000.jpg"

The script prints the five lumbar level predictions (pixel coordinates
and confidence) and saves an annotated image into outputs/.
The original image is never modified.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

from config import OUTPUT_DIR
from utils import (
    LEVELS,
    MEDICAL_DISCLAIMER,
    draw_landmarks,
    load_model,
    norm_to_pixels,
    predict,
    preprocess_image,
)


def load_ground_truth(image_path: Path):
    """
    Look up ground-truth landmarks for this image (if the filename
    appears in the annotation CSV). Returns None if not found.
    """
    csv_path = Path("dataset/coords_pretrain.csv")

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    filename = image_path.name

    df["filename"] = df["filename"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip()

    rows = df[df["filename"] == filename]

    if rows.empty:
        return None

    coords = []

    for level in LEVELS:

        row = rows[rows["level"] == level]

        if row.empty:
            coords.extend([0.0, 0.0])
        else:
            coords.extend(
                [
                    float(row.iloc[0]["relative_x"]),
                    float(row.iloc[0]["relative_y"]),
                ]
            )

    return coords


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Predict lumbar landmarks on a single MRI image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the MRI image (JPG/PNG).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the model checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the annotated image (default: outputs/).",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    try:
        model, device, checkpoint_path = load_model(args.checkpoint)
    except FileNotFoundError as exc:
        print(exc)
        print("Please train the model first: python train.py")
        return

    print(f"Device      : {device}")
    print(f"Checkpoint  : {checkpoint_path}")

    # ---------------------------------------------------------
    # Preprocess + inference
    # ---------------------------------------------------------

    original = Image.open(image_path)

    image_tensor = preprocess_image(original)

    outputs = predict(model, image_tensor, device)

    coords = outputs["coords"].cpu().numpy().reshape(-1)
    confidence = outputs["confidence"].cpu().numpy().reshape(-1)

    width, height = original.size

    pixels = norm_to_pixels(coords, width, height)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print()
    print(f"Image       : {image_path.name}")
    print(f"Resolution  : {width} x {height}")
    print()

    for i, level in enumerate(LEVELS):
        print(level)
        print(f"X           : {pixels[i, 0]:.1f}")
        print(f"Y           : {pixels[i, 1]:.1f}")
        print(f"Confidence  : {confidence[i]:.3f}")
        print()

    print(MEDICAL_DISCLAIMER)

    # ---------------------------------------------------------
    # Visualize + save
    # ---------------------------------------------------------

    ground_truth = load_ground_truth(image_path)

    annotated = draw_landmarks(
        original,
        coords=coords,
        confidence=confidence,
        ground_truth=ground_truth,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"{image_path.stem}_{stamp}_annotated.jpg"

    annotated.save(output_path)

    print()
    print(f"Annotated image saved to : {output_path}")


if __name__ == "__main__":

    main()
