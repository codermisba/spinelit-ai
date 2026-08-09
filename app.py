"""
app.py
======

Simple local Gradio UI for the Spine Foundation Model.

Launches at http://127.0.0.1:7860

Usage
-----
python app.py

This is a research prototype for lumbar spine landmark localization only.
It does NOT provide any medical diagnosis.
"""

import pandas as pd
import gradio as gr

from config import BEST_MODEL
from utils import (
    LEVELS,
    MEDICAL_DISCLAIMER,
    draw_landmarks,
    load_model,
    norm_to_pixels,
    predict,
    preprocess_image,
)

# ---------------------------------------------------------
# Load model at startup (do not crash if checkpoint is missing)
# ---------------------------------------------------------

MODEL = None
DEVICE = None
CHECKPOINT = BEST_MODEL

try:
    MODEL, DEVICE, CHECKPOINT = load_model(BEST_MODEL)
    print(f"Model loaded from : {CHECKPOINT}")
    print(f"Device            : {DEVICE}")
except FileNotFoundError as exc:
    print(exc)
    print("Model NOT loaded. Please train the model first.")


def build_result_table(coords_px, confidence) -> pd.DataFrame:
    """Build the Level | X | Y | Confidence table."""
    return pd.DataFrame(
        {
            "Level": LEVELS,
            "X": [round(float(x), 1) for x in coords_px[:, 0]],
            "Y": [round(float(y), 1) for y in coords_px[:, 1]],
            "Confidence": [round(float(c), 3) for c in confidence],
        }
    )


def analyze(image):
    """Run the full pipeline: preprocess -> infer -> visualize -> table."""

    status = f"{MEDICAL_DISCLAIMER}"

    if image is None:
        return None, None, "Please upload an MRI image first. " + status

    if MODEL is None:
        return (
            None,
            None,
            "No trained model checkpoint found. "
            "Please train the model first. " + status,
        )

    image_tensor = preprocess_image(image)

    outputs = predict(MODEL, image_tensor, DEVICE)

    coords = outputs["coords"].cpu().numpy().reshape(-1)
    confidence = outputs["confidence"].cpu().numpy().reshape(-1)

    width, height = image.size

    pixels = norm_to_pixels(coords, width, height)

    annotated = draw_landmarks(
        image,
        coords=coords,
        confidence=confidence,
    )

    table = build_result_table(pixels, confidence)

    status = (
        f"Analysis complete. Landmarks localized in 5 lumbar levels. "
        f"{status}"
    )

    return annotated, table, status


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

with gr.Blocks(title="Spine Foundation Model") as demo:

    gr.Markdown(
        """
        # Spine Foundation Model

        **AI-assisted lumbar spine landmark localization**

        Upload an MRI image and the model will localize the five lumbar
        levels (L1/L2, L2/L3, L3/L4, L4/L5, L5/S1).

        > Research prototype — not a medical diagnostic tool.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(
                type="pil",
                label="MRI Image",
            )

            analyze_button = gr.Button(
                "Analyze",
                variant="primary",
            )

        with gr.Column(scale=1):

            image_output = gr.Image(
                type="pil",
                label="Annotated Landmarks",
            )

            table_output = gr.Dataframe(
                headers=["Level", "X", "Y", "Confidence"],
                datatype=["str", "number", "number", "number"],
                label="Prediction Table",
            )

            status_output = gr.Markdown()

    status_text = (
        "**Model Status:** "
        + ("Loaded" if MODEL is not None else "Not Loaded")
        + (
            " — No trained model checkpoint found. "
            "Please train the model first."
            if MODEL is None
            else ""
        )
    )

    gr.Markdown(
        f"""
        ---
        {status_text}

        **Checkpoint:** {CHECKPOINT.name}

        {MEDICAL_DISCLAIMER}
        """
    )

    analyze_button.click(
        fn=analyze,
        inputs=image_input,
        outputs=[image_output, table_output, status_output],
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the Spine Foundation Model UI."
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public URL (needed when running on Google Colab).",
    )

    args = parser.parse_args()

    demo.launch(share=args.share)
