"""
cli_pipeline.py
===============

Command-line entry for the agentic spine pipeline.

Usage
-----
python cli_pipeline.py --image scan.jpg \
    --age 58 --sex female --pain-scale 6 --modality mri \
    --pain-years 4 --start-year 2022 --symptoms "lower back pain..." \
    --years-ahead 5

Prints the full PipelineResult as JSON (or a readable text summary
with --pretty). Saves the annotated image into outputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schemas import PatientInput, PipelineResult
from orchestrator import AgenticPipeline


def _pp(result: PipelineResult) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("AGENTIC SPINE PIPELINE RESULT")
    lines.append("=" * 70)
    lines.append(f"patient: {result.patient}")
    lines.append(f"vision:  {result.evidence and result.evidence.image_processed}")
    lines.append("--- symptoms ---")
    lines.append(json.dumps(result.symptoms.model_dump() if result.symptoms else {}))
    lines.append("--- interpretation ---")
    lines.append(json.dumps(result.interpretation.model_dump() if result.interpretation
                            else {}))
    lines.append("--- verification ---")
    lines.append(json.dumps(result.verification.model_dump() if result.verification
                            else {}))
    lines.append("--- longitudinal ---")
    lines.append(json.dumps(result.longitudinal.model_dump() if result.longitudinal
                            else {}))
    lines.append("--- report ---")
    lines.append(json.dumps(result.report.model_dump() if result.report else {}))
    lines.append("--- annotated image ---")
    lines.append(result.annotated_image_path or "(none)")
    lines.append("--- errors ---")
    lines += [f"- {e}" for e in result.errors] or ["(none)"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic spine pipeline.")
    parser.add_argument("--image", required=True, help="Spine image path.")
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", type=str, default=None,
                        choices=["male", "female"])
    parser.add_argument("--pain-scale", type=float, default=None)
    parser.add_argument("--modality", type=str, default=None,
                        choices=["xray", "mri"])
    parser.add_argument("--pain-years", type=float, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--symptoms", type=str, default="")
    parser.add_argument("--years-ahead", type=float, default=5.0)
    parser.add_argument("--pretty", action="store_true",
                        help="print a readable summary instead of raw JSON")
    parser.add_argument("--provider", type=str, default=None,
                        help="llm provider: gemini | ollama")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        return

    patient = PatientInput(
        image_path=args.image, age=args.age, sex=args.sex or "",
        pain_scale=args.pain_scale, modality=args.modality or "",
        pain_years=args.pain_years, start_year=args.start_year,
        symptoms_text=args.symptoms, years_ahead=args.years_ahead,
    )
    result = AgenticPipeline().run(patient, provider=args.provider)

    if args.pretty:
        print(_pp(result))
    else:
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
