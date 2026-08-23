"""
app.py
======

Local Gradio UI for the Spine Foundation Model.

Launches at http://127.0.0.1:7860

Usage
-----
python app.py            # local
python app.py --share    # public URL (Colab)

Upload an X-ray/MRI image, optionally fill in patient/clinical data,
and the model reports:
- all lumbar vertebrae (L1-L5) and disc levels with confidence scores
- geometric disc-space / listhesis indicators
- DDD grade estimate + spondylolisthesis estimate with confidence
- longitudinal progression risk when clinical data is provided

Research prototype — not a medical diagnostic tool.
"""

import pandas as pd
import gradio as gr
import torch

from config import (
    BEST_MODEL,
    DISC_LEVELS,
    LONGITUDINAL_MODEL_NAME,
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

MODEL = None
DEVICE = None
CHECKPOINT = BEST_MODEL
RISK_MODEL = None

try:
    MODEL, DEVICE, CHECKPOINT = load_model(BEST_MODEL)
    print(f"Model loaded from : {CHECKPOINT}")
except FileNotFoundError as exc:
    print(exc)

risk_path = LONGITUDINAL_MODEL_NAME
if risk_path.exists():
    RISK_MODEL = LongitudinalRiskModel()
    state = torch.load(risk_path, map_location="cpu", weights_only=False)
    RISK_MODEL.load_state_dict(state["model_state_dict"])
    RISK_MODEL.eval()
    print(f"Longitudinal model loaded from : {risk_path}")


def analyze(image, age, sex, pain_scale, modality, pain_years, start_year,
            years_ahead):
    """Full pipeline: imaging inference -> tables -> optional risk model."""

    if image is None:
        return None, None, None, None, None, "Please upload an image first."

    if MODEL is None:
        return (None, None, None, None, None,
                "No trained checkpoint found. Run python train.py first.")

    outputs = predict(MODEL, preprocess_image(image), DEVICE)
    decoded = decode_outputs(outputs)

    width, height = image.size
    pixels = norm_to_pixels(decoded["points"], width, height)

    annotated = draw_landmarks(
        image,
        coords=decoded["points"],
        confidence=decoded["localization_conf"],
    )

    n_v = len(VERTEBRAE)

    landmark_table = pd.DataFrame(
        {
            "Point": VERTEBRAE + DISC_LEVELS,
            "Type": ["Vertebra"] * n_v + ["Disc"] * len(DISC_LEVELS),
            "X": [round(float(p[0]), 1) for p in pixels],
            "Y": [round(float(p[1]), 1) for p in pixels],
            "Confidence": [round(float(c), 3)
                           for c in decoded["localization_conf"]],
        }
    )

    geo = compute_geometric_indicators(pixels)
    geometry_table = pd.DataFrame(
        {
            "Level": [g["level"] for g in geo],
            "Relative Disc Space": [g["relative_space"] for g in geo],
            "Space Narrowed": ["YES" if g["narrowed"] else "-" for g in geo],
            "Centre Offset Ratio": [g["offset_ratio"] for g in geo],
            "Listhesis Possible": ["YES" if g["offset_flag"] else "-"
                                   for g in geo],
        }
    ) if geo else pd.DataFrame()

    ddd_table = pd.DataFrame(
        {
            "Level": DISC_LEVELS,
            "DDD Grade (0-4)": [round(float(g), 2)
                                for g in decoded["ddd_grade"]],
            "Severity": [severity_from_grade(g)
                         for g in decoded["ddd_grade"]],
            "Confidence": [round(float(c), 3) for c in decoded["ddd_conf"]],
        }
    )

    spondy_table = pd.DataFrame(
        {
            "Level": DISC_LEVELS,
            "Slip %": [round(float(s), 1)
                       for s in decoded["spondy_slip_pct"]],
            "Meyerding": [meyerding_grade(s)
                          for s in decoded["spondy_slip_pct"]],
            "Confidence": [round(float(c), 3)
                           for c in decoded["spondy_conf"]],
        }
    )

    clinical_complete = all(
        v is not None and v != ""
        for v in [age, sex, pain_scale, modality, pain_years, start_year]
    )

    status = MEDICAL_DISCLAIMER

    if clinical_complete and RISK_MODEL is not None:
        results = predict_risk(
            RISK_MODEL,
            ClinicalInputs(
                age=int(age), sex=str(sex), pain_scale=float(pain_scale),
                modality=str(modality).lower(),
                pain_years=float(pain_years),
                start_year=int(start_year),
            ),
            image_features=outputs["features"][0].float().cpu(),
            years_ahead=float(years_ahead),
        )
        risk_rows = [{"Level": level,
                      "Predicted Grade": round(results["future_grade"][i], 2),
                      "Worsening Risk": f"{results['progression_risk'][i]*100:.1f}%"}
                     for i, level in enumerate(DISC_LEVELS)]
        risk_rows.append({"Level": "OVERALL",
                          "Predicted Grade": "-",
                          "Worsening Risk":
                              f"{results['overall_risk']*100:.1f}%"})
        ddd_table = pd.concat(
            [ddd_table, pd.DataFrame(risk_rows)], axis=1
        )
        status = (
            f"Longitudinal prediction (+{years_ahead:g} yr horizon): "
            f"overall worsening risk {results['overall_risk']*100:.1f}%. "
            + status
        )
    elif clinical_complete:
        status = (
            "Clinical inputs received but the longitudinal model is not "
            "trained yet (run `python train_clinical.py`). " + status
        )

    return (annotated, landmark_table, geometry_table, ddd_table,
            spondy_table, status)


with gr.Blocks(title="Spine Foundation Model") as demo:

    gr.Markdown(
        """
        # Spine Foundation Model

        **AI-assisted lumbar spine analysis — X-ray & MRI**

        Detects all lumbar vertebrae (L1-L5) and disc levels (L1/L2 - L5/S1)
        with confidence scores, estimates disc degeneration and
        spondylolisthesis, and predicts longitudinal progression risk.

        > Research prototype — not a medical diagnostic tool.
        """
    )

    with gr.Row():

        with gr.Column(scale=1):

            image_input = gr.Image(type="pil", label="X-ray / MRI Image")

            gr.Markdown("### Patient / Clinical Information")
            gr.Markdown("*Used for the longitudinal risk prediction.*")

            with gr.Row():
                age_input = gr.Number(label="Age (years)", precision=0)
                sex_input = gr.Radio(["male", "female"], label="Sex")

            pain_input = gr.Slider(0, 10, step=1, label="Pain scale (VAS)")

            modality_input = gr.Radio(["X-ray", "MRI"],
                                      label="Imaging done")

            with gr.Row():
                pain_years_input = gr.Slider(0, 50, step=1,
                                             label="Years in pain")
                start_year_input = gr.Number(label="Symptom start year",
                                             precision=0)

            horizon_input = gr.Slider(1, 10, step=1, value=5,
                                      label="Predict years ahead")

            analyze_button = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):

            image_output = gr.Image(type="pil",
                                    label="Detected Vertebrae & Discs")

            gr.Markdown("### Landmarks (all vertebrae + discs)")
            landmark_output = gr.Dataframe(label="Landmarks & Confidence")

            gr.Markdown("### Geometric Indicators")
            geometry_output = gr.Dataframe()

    with gr.Row():

        with gr.Column():
            gr.Markdown("### Disc Degeneration (DDD)")
            ddd_output = gr.Dataframe()

        with gr.Column():
            gr.Markdown("### Spondylolisthesis")
            spondy_output = gr.Dataframe()

    status_output = gr.Markdown()

    analyze_button.click(
        fn=analyze,
        inputs=[image_input, age_input, sex_input, pain_input,
                modality_input, pain_years_input, start_year_input,
                horizon_input],
        outputs=[image_output, landmark_output, geometry_output,
                 ddd_output, spondy_output, status_output],
    )

    gr.Markdown(
        f"""
        ---
        **Model Status:** {"Loaded" if MODEL is not None else "Not Loaded"}
        | **Checkpoint:** {CHECKPOINT.name}
        | Longitudinal model: {"Loaded" if RISK_MODEL is not None else "Not trained"}
        | DDD labels: {"available" if ddd_labels_available() else "not provided"}
        | Spondy labels: {"available" if spondy_labels_available() else "not provided"}

        {MEDICAL_DISCLAIMER}
        """
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo.launch(share=args.share)
