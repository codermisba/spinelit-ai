"""
decision_support.py
===================

End-to-end orchestration of the decision-support pipeline (spec §22,
modules 9-16). This module ties together:

    contradiction engine (9)  -> evidence fusion (10) -> clinical
    correlation (11)          -> longitudinal-risk validation (12)
    -> structured JSON (13)   -> LLM report (14) -> output validation (15)
    -> audit logging (16)

Compared to the existing agentic pipeline (`orchestrator.py`) this is the
deterministic, safety-first path: the LLM only ever sees validated
structured evidence, cannot override it, and its output is validated /
regenerated / replaced by a conservative fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pipeline_config import PipelineConfig
from evidence_schemas import DecisionSupportResult
from structured_builder import (
    RawCaseInput,
    RawDDDPrediction,
    RawGeometry,
    RawImageQuality,
    RawSpondyPrediction,
    build_structured_case,
)
from pipeline_report import generate_report
from audit_log import AuditLogger, build_audit_summary


def run_decision_support(
    case_id: str,
    *,
    ddd_predictions=None,
    spondy_predictions=None,
    geometry=None,
    image_quality=None,
    clinical_context=None,
    symptoms=None,
    longitudinal_horizon: int = 5,
    pipeline_version: str | None = None,
    provider: str | None = None,
    config: Optional[PipelineConfig] = None,
    use_llm: bool = True,
) -> DecisionSupportResult:
    """Run the full decision-support pipeline for one case.

    Inputs are plain dicts/dataclasses (see structured_builder.Raw* types).
    """
    cfg = config or PipelineConfig.get()
    version = pipeline_version or cfg.pipeline.version

    raw = RawCaseInput(
        case_id=case_id,
        image_quality=_to_raw_image_quality(image_quality),
        ddd_predictions=_to_raw_ddd(ddd_predictions),
        spondy_predictions=_to_raw_spondy(spondy_predictions),
        geometry=_to_raw_geometry(geometry),
        clinical_context=dict(clinical_context or {}),
        symptoms=dict(symptoms or {}),
        longitudinal_horizon=longitudinal_horizon,
    )
    case = build_structured_case(raw, config=cfg)

    if use_llm:
        result = generate_report(case, config=cfg, provider=provider)
    else:
        from pipeline_report import ReportGenerator
        result = ReportGenerator(cfg, provider).generate(
            case, attempt_llm=False)

    # -- audit logging (module 16) --------------------------------
    logger = AuditLogger(cfg)
    run_id = logger.new_run_id()
    summary = build_audit_summary(case, result.report)
    logger.log(
        run_id=run_id,
        case_id=case_id,
        pipeline_version=version,
        input_identifiers={"case_id": case_id},
        model_versions={
            "pipeline": version,
            "contradiction_engine": "deterministic-v1",
            "evidence_fusion": "weighted-matrix-v1",
        },
        summary=summary,
    )
    result.audit_ids.append(run_id)
    return result


# ------------------------------------------------------------------
# Input coercion helpers (accept plain dicts AND Raw dataclasses)
# ------------------------------------------------------------------

def _to_raw_image_quality(v) -> Optional[RawImageQuality]:
    if v is None:
        return None
    if isinstance(v, RawImageQuality):
        return v
    return RawImageQuality(
        status=str(v.get("status", "adequate")),
        confidence=v.get("confidence"),
        issues=list(v.get("issues", [])),
    )


def _to_raw_ddd(items) -> list[RawDDDPrediction]:
    from structured_builder import RawDDDPrediction as R
    out = []
    for item in items or []:
        if isinstance(item, R):
            out.append(item)
            continue
        out.append(R(
            level=str(item["level"] or item.get("level")),
            pfirrmann_grade=item.get("pfirrmann_grade"),
            raw_confidence=item.get("raw_confidence"),
            calibrated_probability=item.get("calibrated_probability"),
            model_version=str(item.get("model_version", "")),
        ))
    return out


def _to_raw_spondy(items) -> list[RawSpondyPrediction]:
    from structured_builder import RawSpondyPrediction as R
    out = []
    for item in items or []:
        if isinstance(item, R):
            out.append(item)
            continue
        out.append(R(
            level=str(item.get("level")),
            slip_percent=item.get("slip_percent"),
            grade_label=str(item.get("grade_label", "")),
            raw_confidence=item.get("raw_confidence"),
            calibrated_probability=item.get("calibrated_probability"),
            model_version=str(item.get("model_version", "")),
        ))
    return out


def _to_raw_geometry(items) -> list[RawGeometry]:
    from structured_builder import RawGeometry as R
    out = []
    for item in items or []:
        if isinstance(item, R):
            out.append(item)
            continue
        out.append(R(
            level=str(item.get("level")),
            offset_ratio=item.get("offset_ratio"),
            listhesis_possible=item.get("listhesis_possible"),
            relative_space=item.get("relative_space"),
            space_narrowed=item.get("space_narrowed"),
            landmark_confidence=item.get("landmark_confidence"),
            measurement_quality=item.get("measurement_quality"),
        ))
    return out


# ------------------------------------------------------------------
# Convenience constructors for the worked example
# ------------------------------------------------------------------

def example_case_geometry() -> list[dict]:
    """Independent geometric measurements from the worked example."""
    return [
        {"level": "L1/L2", "offset_ratio": 0.270, "listhesis_possible": True},
        {"level": "L2/L3", "offset_ratio": 0.069, "listhesis_possible": False},
        {"level": "L3/L4", "offset_ratio": 0.096, "listhesis_possible": False},
        {"level": "L4/L5", "offset_ratio": 0.041, "listhesis_possible": False,
         "relative_space": 0.734, "space_narrowed": True},
    ]


def example_case_spondy() -> list[dict]:
    return [
        {"level": "L1/L2", "slip_percent": 50.1, "grade_label": "grade III",
         "raw_confidence": 0.579, "calibrated_probability": 0.742},
        {"level": "L2/L3", "slip_percent": 51.5, "grade_label": "grade III",
         "calibrated_probability": 0.739},
        {"level": "L3/L4", "slip_percent": 46.0, "grade_label": "grade III",
         "raw_confidence": 0.499, "calibrated_probability": 0.686},
        {"level": "L4/L5", "slip_percent": 46.8, "grade_label": "grade III",
         "raw_confidence": 0.472, "calibrated_probability": 0.679},
        {"level": "L5/S1", "slip_percent": 50.1, "grade_label": "grade III",
         "calibrated_probability": 0.730},
    ]


def example_case_ddd() -> list[dict]:
    return [
        {"level": "L1/L2", "pfirrmann_grade": 2.158, "raw_confidence": 0.441},
        {"level": "L2/L3", "pfirrmann_grade": 1.848, "raw_confidence": 0.463},
        {"level": "L3/L4", "pfirrmann_grade": 2.054, "raw_confidence": 0.508},
        {"level": "L4/L5", "pfirrmann_grade": 1.889, "raw_confidence": 0.460},
        {"level": "L5/S1", "pfirrmann_grade": 2.000, "raw_confidence": 0.459},
    ]


def example_clinical_context() -> dict:
    return {
        "age": 52,
        "pain_score": 5.0,
        "pain_duration_years": 4,
    }


def example_case(case_id: str = "example-01") -> RawCaseInput:
    """Assemble the full worked example (§21 / §29) as RawCaseInput."""
    return RawCaseInput(
        case_id=case_id,
        image_quality=RawImageQuality(status="adequate"),
        ddd_predictions=_to_raw_ddd(example_case_ddd()),
        spondy_predictions=_to_raw_spondy(example_case_spondy()),
        geometry=_to_raw_geometry(example_case_geometry()),
        clinical_context=example_clinical_context(),
        symptoms={"right_leg_numbness": True},
        longitudinal_horizon=5,
    )


def run_example(use_llm: bool = True) -> DecisionSupportResult:
    return run_decision_support(
        "example-01",
        ddd_predictions=example_case_ddd(),
        spondy_predictions=example_case_spondy(),
        geometry=example_case_geometry(),
        clinical_context=example_clinical_context(),
        symptoms={"right_leg_numbness": True},
        use_llm=use_llm,
    )