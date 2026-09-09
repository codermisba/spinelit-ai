"""
cli_decision_support.py
=======================

Command-line entry for the deterministic decision-support pipeline.

Examples
--------
# Worked example case (spec §21), full pipeline + report:
python cli_decision_support.py --example --pretty

# Worked example with a real LLM call (Gemini):
python cli_decision_support.py --example --pretty --provider gemini

# Deterministic only (no LLM; conservative fallback report):
python cli_decision_support.py --example --pretty --no-llm
"""

from __future__ import annotations

import argparse
import json
import sys

from decision_support import run_decision_support, run_example


def _pp(result) -> str:
    from pipeline_report import SYSTEM_PROMPT  # noqa: F401
    lines = []
    lines.append("=" * 74)
    lines.append("SPINE DECISION-SUPPORT RESULT")
    lines.append("=" * 74)
    lines.append(f"case:            {result.case.case_id}")
    lines.append(f"pipeline version:{result.case.pipeline_version}")
    lines.append(f"image quality:   {result.case.image_quality.status.value}")
    lines.append(f"overall status:  {result.case.overall_status.value}")
    lines.append(f"requires review: {result.case.requires_radiologist_review}")
    lines.append(f"LLM used:        {result.llm_used}")
    lines.append("")
    lines.append("--- findings ---")
    for f in result.case.findings:
        lines.append(
            f"[{f.finding_type.value}] {f.level} | conf={f.confidence_status.value} "
            f"| evid={f.evidence_status.value} | contradiction="
            f"{f.contradiction_severity.value}"
        )
        lines.append(f"    {f.final_language}")
        lines.append(f"    evidence: {', '.join(f.evidence_ids)}")
    lines.append("")
    lines.append("--- contradictions ---")
    for c in result.case.contradictions:
        lines.append(f"{c.level} [{c.severity.value}]: {c.description}")
    lines.append("")
    lines.append("--- longitudinal ---")
    lines.append(json.dumps(result.case.longitudinal_risk.model_dump(), indent=2))
    lines.append("")
    lines.append("--- report ---")
    if result.report:
        lines.append(result.report.text)
    lines.append("")
    lines.append("--- validation ---")
    lines.append(f"report_validated: {result.report_validated}")
    for e in result.validation_errors:
        lines.append(f"  validation: {e}")
    for e in result.errors:
        lines.append(f"  error: {e}")
    lines.append(f"audit_run_ids: {result.audit_ids}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spine decision-support pipeline (deterministic safety layer).")
    parser.add_argument("--example", action="store_true",
                        help="run the worked example case (spec §21).")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM and produce the conservative fallback report.")
    parser.add_argument("--provider", type=str, default=None,
                        help="llm provider: gemini | ollama (default: config).")
    parser.add_argument("--pretty", action="store_true",
                        help="human-readable summary instead of raw JSON.")
    parser.add_argument("--out", type=str, default=None,
                        help="write the structured JSON to a file.")
    args = parser.parse_args()

    if args.example:
        result = run_example(use_llm=not args.no_llm)
    else:
        parser.print_help()
        sys.exit(0)

    if args.pretty:
        print(_pp(result))
    else:
        print(json.dumps(result.to_dict(), indent=2, default=str))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        print(f"\nwrote structured JSON -> {args.out}")


if __name__ == "__main__":
    main()