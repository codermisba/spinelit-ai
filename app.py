"""
app.py
======

Gradio UI for the end-to-end *agentic* spine pipeline.

For one case the user provides:
  - a spine X-ray / MRI image
  - clinical/patient fields (optional)
  - free-text symptoms / complaints (optional)

The Agent Orchestrator runs: symptom extraction -> vision engine
(calibrated findings) -> clinical-vision fusion -> verification/critique
-> longitudinal outlook -> radiology-style report.

The UI shows: the annotated image, structured finding tables (per level,
with calibrated probabilities + raw confidence), the verification audit,
the longitudinal outlook and the written report.

Default: python app.py            (local  http://127.0.0.1:7860)
        python app.py --share    (public link)
"""

import pandas as pd
import gradio as gr

from config import DISC_LEVELS, GEMINI_API_KEY, LLM_PROVIDER
from schemas import PatientInput
from orchestrator import AgenticPipeline

MODEL_STATUS_HEADER = (
    f"LLM backend: **{LLM_PROVIDER}** | "
    f"API key: {'set' if GEMINI_API_KEY else 'NOT SET (set GEMINI_API_KEY)'}"
)


def _build_bundle() -> tuple[AgenticPipeline, dict]:
    vision = None
    try:
        vision = AgenticPipeline().vision
    except Exception:  # noqa: BLE001
        vision = None
    status = vision.status() if vision else {"loaded": False}
    return vision, status


def _ddd_rows(result):
    if not result.evidence or not result.evidence.ddd:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Level": f.level,
            "Grade (0-4)": f.grade,
            "Severity": f.severity,
            "P(DDD)": f.calibrated_probability,
            "Raw Conf": f.raw_confidence,
            "Loc Qual": f.localization_quality,
        }
        for f in result.evidence.ddd
    ])


def _spondy_rows(result):
    if not result.evidence or not result.evidence.spondy:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Level": f.level,
            "Slip %": f.slip_percent,
            "Meyerding": f.meyerding,
            "P(Spondy)": f.calibrated_probability,
            "Raw Conf": f.raw_confidence,
            "Loc Qual": f.localization_quality,
        }
        for f in result.evidence.spondy
    ])


def _geo_rows(result):
    if not result.evidence or not result.evidence.geometric_indicators:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "Level": g.get("level"),
            "Relative Space": g.get("relative_space"),
            "Narrowed": "YES" if g.get("narrowed") else "-",
            "Offset Ratio": g.get("offset_ratio"),
            "Listhesis Possible": "YES" if g.get("offset_flag") else "-",
        }
        for g in result.evidence.geometric_indicators
    ])


def _verification_text(result):
    v = result.verification
    if v is None:
        return "No verification available."
    lines = [f"**Overall status:** {v.overall_status}"]
    if v.summary:
        lines.append(v.summary)
    for it in v.items:
        lines.append(
            f"- {it.status.upper()}: {it.claim} "
            f"(agree {it.agree_probability:.2f}). "
            f"Supports: {it.supports or '-'}. "
            f"Conflicts: {it.conflicts or '-'}."
            + (f" **Correction:** {it.correction}" if it.correction else "")
        )
    return "\n\n".join(lines)


def _report_text(result):
    r = result.report
    if r is None:
        return "No written report produced (LLM not configured). See tables."
    parts = []
    if r.findings:
        parts.append("### Findings\n" + r.findings)
    if r.diagnoses:
        rows = []
        for d in r.diagnoses:
            rows.append(
                f"- **{d.get('disease')}** — P={d.get('calibrated_probability')}: "
                f"{d.get('assessment')}. Evidence: {', '.join(d.get('evidence', []))}"
            )
        parts.append("### Diagnoses\n" + "\n".join(rows))
    if r.impression:
        parts.append("### Impression\n" + r.impression)
    if r.confidence_statement:
        parts.append("### Confidence\n" + r.confidence_statement)
    if r.recommendations:
        parts.append("### Recommendations\n" + "\n".join(
            f"- {x}" for x in r.recommendations))
    if r.caveats:
        parts.append("### Caveats\n" + "\n".join(f"- {x}" for x in r.caveats))
    return "\n\n".join(parts)


def _longitudinal_text(result):
    lo = result.longitudinal
    if lo is None:
        return "No longitudinal assessment produced."
    text = (f"Horizon: +{lo.years_ahead} years | "
            f"Overall worsening risk: {lo.overall_risk:.2f}")
    if lo.per_level_future_grade:
        text += "\n\n**Per-level** (future grade, worsening risk):\n"
        for i, level in enumerate(DISC_LEVELS):
            fg = lo.per_level_future_grade[i] if i < len(lo.per_level_future_grade) else None
            wr = lo.per_level_worsening_risk[i] if i < len(lo.per_level_worsening_risk) else None
            text += (f"- {level}: future grade = {fg}, "
                     f"worsening risk = {wr}\n")
    if lo.narrative:
        text += "\n\n**Narrative:**\n" + lo.narrative
    return text


def _errors_text(result):
    if not result.errors:
        return "No pipeline errors."
    return "\n".join(f"- {e}" for e in result.errors)


def analyze(image, age, sex, pain_scale, modality, pain_years, start_year,
            symptoms_text, years_ahead):
    pipeline = AgenticPipeline()
    patient = PatientInput(
        image_path="", age=int(age) if age else None,
        sex=str(sex) if sex else "",
        pain_scale=float(pain_scale) if pain_scale is not None else None,
        modality=str(modality).lower() if modality else "",
        pain_years=float(pain_years) if pain_years is not None else None,
        start_year=int(start_year) if start_year else None,
        symptoms_text=str(symptoms_text or ""),
        years_ahead=float(years_ahead) if years_ahead else 5.0,
    )
    result = pipeline.run(patient, pil_image=image)

    annotated = None
    if result.annotated_image_path:
        try:
            from PIL import Image
            annotated = Image.open(result.annotated_image_path)
        except Exception:  # noqa: BLE001
            annotated = None

    return (
        annotated,
        _ddd_rows(result),
        _spondy_rows(result),
        _geo_rows(result),
        _verification_text(result),
        _longitudinal_text(result),
        _report_text(result),
        MODEL_STATUS_HEADER
        + ("\n\n### Pipeline errors\n" + _errors_text(result)
           if result.errors else ""),
    )


with gr.Blocks(title="Agentic Spine AI Pipeline") as demo:
    gr.Markdown(
        """
        # Agentic AI Pipeline — Lumbar Disc Degenerative Disease &
        Spondylolisthesis

        An end-to-end **agentic** pipeline: free-text symptoms are parsed by an
        LLM intake agent, the spine image is analysed by a calibrated vision
        engine, a reasoning agent fuses clinical + imaging evidence, a
        **verification/critique agent** audits the interpretation for accuracy,
        a longitudinal agent assesses progression, and a report agent writes a
        radiologist-style report — every finding carries a calibrated
        probability and cited evidence.

        > Research aid — not a medical diagnostic tool.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Spine X-ray / MRI image")

            gr.Markdown("### Clinical record (optional)")
            with gr.Row():
                age_input = gr.Number(label="Age", precision=0)
                sex_input = gr.Radio(["male", "female"], label="Sex")
            pain_input = gr.Slider(0, 10, step=0.5, value=5,
                                   label="Pain scale (VAS 0-10)")
            modality_input = gr.Radio(["X-ray", "MRI"], label="Imaging modality")
            with gr.Row():
                pain_years_input = gr.Number(label="Years in pain", precision=0)
                start_year_input = gr.Number(label="Symptom start year",
                                             precision=0)
            horizon_input = gr.Slider(1, 10, step=1, value=5,
                                      label="Longitudinal horizon (years)")

            gr.Markdown("### Free-text symptoms (optional)")
            symptoms_input = gr.Textbox(
                lines=5, placeholder=
                "e.g. 'Lower back pain for 2 years, worse with sitting. "
                "Numbness and tingling down the back of my right leg...'",
                label="Patient's own words",
            )

            analyze_button = gr.Button("Run Agentic Pipeline",
                                       variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(type="pil",
                                    label="Annotated spine image")
            with gr.Row():
                ddd_output = gr.Dataframe(label="Disc Degenerative Disease")
                spondy_output = gr.Dataframe(label="Spondylolisthesis")
            geo_output = gr.Dataframe(label="Geometric indicators")
            verification_output = gr.Markdown(label="Verification audit")
            longitudinal_output = gr.Markdown(label="Longitudinal outlook")
            report_output = gr.Markdown(label="Radiologist-style report")
            status_output = gr.Markdown()

    analyze_button.click(
        fn=analyze,
        inputs=[image_input, age_input, sex_input, pain_input,
                modality_input, pain_years_input, start_year_input,
                symptoms_input, horizon_input],
        outputs=[image_output, ddd_output, spondy_output, geo_output,
                 verification_output, longitudinal_output, report_output,
                 status_output],
    )

    gr.Markdown(MODEL_STATUS_HEADER)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(share=args.share)
