"""
pipeline_report.py
==================

Final LLM report generation (spec §18–§21).

The LLM receives ONLY the validated `StructuredCase` (structured evidence).
It is forbidden from inventing findings, upgrading uncertainty, or
overriding Levels 1-4 (§25). The system prompt is the conservative
reporting role prompt from spec §19.

Flow:
    1. Serialize the StructuredCase to JSON (schema §15).
    2. Ask the configured LLM (Gemini) for a DecisionSupportReport using
       Gemini structured-output (responseSchema) via llm.py.
    3. Validate the parsed output against the pydantic schema.
    4. Run ReportValidator. If invalid, regenerate up to
       `reporting.llm_max_retries` times.
    5. If the LLM is unavailable / output repeatedly invalid, emit a
       deterministic conservative fallback report built entirely from
       the structured findings (traceable, safe). The fallback can never
       hallucinate because it only renders fields that already exist.
"""

from __future__ import annotations

import json
from typing import Optional

from pipeline_config import PipelineConfig
from evidence_schemas import (
    ConfidenceStatus,
    DecisionSupportReport,
    DecisionSupportResult,
    EvidenceStatus,
    FindingType,
    StructuredCase,
    TraceStatement,
)
from report_validator import ReportValidator
from longitudinal_risk import longitudinal_outlook_sentence

# ------------------------------------------------------------------
# System prompt (spec §19) — used verbatim.
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a conservative medical-imaging decision-support report generator.\n"
    "\n"
    "You are not a physician and must not provide an autonomous diagnosis.\n"
    "\n"
    "Your task is to summarize structured evidence produced by an MRI analysis "
    "pipeline.\n"
    "\n"
    "Follow the evidence hierarchy:\n"
    "1. Image/anatomical validity\n"
    "2. Independent geometric validation\n"
    "3. Model predictions\n"
    "4. Clinical context\n"
    "5. Conservative synthesis\n"
    "\n"
    "Never allow a model prediction to override contradictory independent "
    "geometric evidence.\n"
    "\n"
    "Never convert a model probability directly into diagnostic certainty.\n"
    "\n"
    "When raw model confidence is below the configured threshold, explicitly "
    "disclose the low confidence.\n"
    "\n"
    "When model and geometric evidence disagree, explicitly describe the "
    "contradiction and downgrade the finding.\n"
    "\n"
    "Do not label a finding as confirmed unless the structured evidence "
    "explicitly marks it as confirmed.\n"
    "\n"
    "Do not call a finding an artifact unless artifact status is independently "
    "established.\n"
    "\n"
    "Use terms such as 'not corroborated', 'indeterminate', 'provisional', or "
    "'requires specialist review' when appropriate.\n"
    "\n"
    "Keep objective imaging findings separate from symptoms.\n"
    "\n"
    "Do not infer causation from correlation.\n"
    "\n"
    "Do not identify a compressed nerve root unless the structured evidence "
    "explicitly supports it.\n"
    "\n"
    "Do not generate a numeric longitudinal risk if no validated longitudinal "
    "model is available.\n"
    "\n"
    "Do not invent missing information.\n"
    "\n"
    "Every important conclusion must be traceable to one or more structured "
    "evidence fields.\n"
    "\n"
    "The final output is a research/decision-support summary and must state "
    "that it requires review by a qualified radiologist/clinician."
)

DISCLAIMER = (
    "This automated analysis is a research/decision-support aid and is not a "
    "medical diagnosis. Findings require review and validation by a qualified "
    "radiologist and appropriate clinical correlation."
)

REPORT_INSTRUCTIONS_TEMPLATE = """\
You are summarizing the STRUCTURED EVIDENCE below. Reproduce the JSON schema EXACTLY.

Use these exact values from the structured evidence where required:

OVERALL STATUS: {overall_status}

Report every finding in `findings`. Quote each finding's `final_language`.
For each finding, list its `finding_id` and `evidence_ids`.
Where `confidence_status` is LOW_CONFIDENCE, state that the automated
confidence is low (under 'model_confidence').
Where `evidence_status` is CONTRADICTED, NOT_CORROBORATED or INDETERMINATE or
where `contradiction` is true, reproduce the contradiction in 'contradictions'
and never present the finding as confirmed.
The 'provisional_findings' section lists findings in CONTHOR with status
INDETERMINATE/PROVISIONAL/CONTRADICTED/NOT_CORROBORATED or low confidence.
'clinical_correlation' must reflect the case's symptom_correlations and must
NOT claim a nerve-root compression unless explicitly supported.
'longitudinal_outlook' must be: {longitudinal_outlook}
Keep objective imaging findings separate from symptoms.
Every bullet and sentence must remain faithful to the structured evidence.
Do not invent measurements, levels, grades, symptoms or recommendations.

STRUCTURED EVIDENCE (JSON):
{structured_json}
"""

REPORT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "overall_status": {"type": "STRING"},
        "objective_imaging_findings": {
            "type": "ARRAY", "items": {"type": "STRING"}},
        "model_confidence": {
            "type": "ARRAY", "items": {"type": "STRING"}},
        "contradictions": {
            "type": "ARRAY", "items": {"type": "STRING"}},
        "clinical_correlation": {"type": "STRING"},
        "provisional_findings": {
            "type": "ARRAY", "items": {"type": "STRING"}},
        "longitudinal_outlook": {"type": "STRING"},
        "impression": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence_limitations": {
            "type": "ARRAY", "items": {"type": "STRING"}},
        "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "disclaimer": {"type": "STRING"},
    },
    "required": [
        "overall_status", "objective_imaging_findings", "model_confidence",
        "contradictions", "clinical_correlation", "provisional_findings",
        "longitudinal_outlook", "impression", "confidence_limitations",
        "recommendations", "disclaimer",
    ],
}


class ReportGenerator:
    """Generates, validates and (if needed) regnerates the report."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 provider: str | None = None):
        self.config = config or PipelineConfig.get()
        self.provider = provider
        self.validator = ReportValidator(self.config)

    # ------------------------------------------------------------------

    def generate(self, case: StructuredCase,
                 attempt_llm: bool = True) -> DecisionSupportResult:
        result = DecisionSupportResult(case=case)
        if not attempt_llm:
            result.report = self._fallback_report(case)
            result.report_attempts = 0
            result.llm_used = False
            self._ensure_traceability(case, result.report)
            check = self.validator.validate(case, result.report)
            result.report_validated = not check.errors
            result.validation_errors = list(check.errors)
            return result

        attempts = 0
        max_retries = max(0, self.config.reporting.llm_max_retries)

        while attempts <= max_retries:
            attempts += 1
            report = self._call_llm(case, result)
            if report is not None:
                self._ensure_traceability(case, report)
                check = self.validator.validate(case, report)
                if check.valid:
                    result.report = report
                    result.report_validated = True
                    result.report_attempts = attempts
                    result.llm_used = True
                    return result
                result.validation_errors = list(check.errors)
            else:
                result.validation_errors.append(
                    "LLM report failed to parse; falling back."
                )
            if attempts > max_retries:
                break

        # Deterministic conservative fallback — cannot hallucinate.
        result.report = self._fallback_report(case)
        result.report_attempts = attempts
        result.llm_used = False
        self._ensure_traceability(case, result.report)
        check = self.validator.validate(case, result.report)
        if not check.valid:
            result.errors.extend(check.errors)  # surfaced but report is safe
        result.report_validated = True if not check.errors else False
        return result

    # ------------------------------------------------------------------
    # Traceability (deterministic; never left to the LLM)
    # ------------------------------------------------------------------

    def _ensure_traceability(self, case: StructuredCase,
                            report: DecisionSupportReport) -> None:
        """Rebuild statement->evidence mapping when the report lacks it.

        The LLM cannot be trusted to supply traceability; every accepted
        report therefore gets one computed from the structured case.
        """
        if report.traceability:
            return
        for f in case.findings:
            if f.final_language:
                report.traceability.append(TraceStatement(
                    statement=f.final_language,
                    evidence_ids=list(f.evidence_ids),
                ))
        for c in case.contradictions:
            report.traceability.append(TraceStatement(
                statement=c.description,
                evidence_ids=[c.contradiction_id],
            ))
        report.traceability.append(TraceStatement(
            statement=report.longitudinal_outlook,
            evidence_ids=[case.longitudinal_risk.status.value],
        ))
        report.traceability.append(TraceStatement(
            statement=report.disclaimer,
            evidence_ids=["POLICY"],
        ))

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, case: StructuredCase,
                  result: DecisionSupportResult) -> Optional[DecisionSupportReport]:
        try:
            from llm import make_client  # lazy to keep module import cheap
            client = make_client(self.provider)
        except Exception as exc:  # noqa: BLE001 - not configured etc.
            result.errors.append(f"LLM unavailable: {exc}")
            return None

        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + REPORT_INSTRUCTIONS_TEMPLATE.format(
                overall_status=case.overall_status.value,
                longitudinal_outlook=longitudinal_outlook_sentence(
                    case.longitudinal_risk),
                structured_json=_case_json(case),
            )
        )

        try:
            raw = client.complete_json(prompt, response_schema=REPORT_RESPONSE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"LLM call failed: {exc}")
            return None

        return _parse_report(raw)

    # ------------------------------------------------------------------
    # Deterministic fallback report (safe, traceable)
    # ------------------------------------------------------------------

    def _fallback_report(self, case: StructuredCase) -> DecisionSupportReport:
        report = DecisionSupportReport(
            overall_status=case.overall_status.value,
            disclaimer=DISCLAIMER,
            longitudinal_outlook=longitudinal_outlook_sentence(
                case.longitudinal_risk),
            clinical_correlation=_clinical_sentence(case),
        )

        # Objective imaging findings: geometric + supported findings only.
        for f in _sorted_findings(case):
            if f.finding_type == FindingType.DISC_SPACE_NARROWING or (
                f.evidence_status in (
                    EvidenceStatus.SUPPORTED,
                    EvidenceStatus.CONFIRMED,
                    EvidenceStatus.LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT,
                )
            ):
                report.objective_imaging_findings.append(f.final_language)

        if not report.objective_imaging_findings:
            # A case where nothing reached the support threshold still gets a
            # self-describing objective section (keeps the report valid and
            # makes the absence of confirmed findings explicit).
            report.objective_imaging_findings.append(
                "No automated structural finding met the independent-support "
                "criteria; all model-derived findings at this level require "
                "specialist review."
            )

        # Model confidence disclosures.
        low = [f for f in case.findings
               if f.confidence_status == ConfidenceStatus.LOW_CONFIDENCE and
               f.raw_confidence is not None]
        for f in low:
            report.model_confidence.append(
                f"{f.level} {f.finding_type.value}: raw confidence "
                f"{f.raw_confidence:.3f} -> LOW_CONFIDENCE."
            )
        if not low:
            report.model_confidence.append(
                "No low-confidence automated predictions were identified."
            )

        # Contradictions.
        for c in case.contradictions:
            report.contradictions.append(
                f"{c.level} ({c.finding_type.value}): {c.description} "
                f"[severity: {c.severity.value}]"
            )
        contradicted_findings = [
            f for f in case.findings
            if f.evidence_status in (
                EvidenceStatus.CONTRADICTED, EvidenceStatus.NOT_CORROBORATED)
        ]
        for f in contradicted_findings:
            report.contradictions.append(f.final_language)

        # Provisional findings.
        for f in _sorted_findings(case):
            if f.evidence_status in (
                EvidenceStatus.INDETERMINATE,
                EvidenceStatus.PROVISIONAL,
                EvidenceStatus.NOT_CORROBORATED,
                EvidenceStatus.LOW_CONFIDENCE_UNSUPPORTED,
            ) or (f.evidence_status == EvidenceStatus.LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT):
                report.provisional_findings.append(
                    f"{f.level} ({f.finding_type.value}): {f.final_language} "
                    f"[evidence: {', '.join(f.evidence_ids)}]"
                )

        # Confidence / limitations.
        report.confidence_limitations = _limitations(case)

        # Conservative recommendations.
        report.recommendations = [
            "This summary is a research/decision-support aid and must not be "
            "used alone to guide management.",
            "Correlation with the complete imaging examination and clinical "
            "neurological assessment is recommended.",
            "Findings with low confidence or contradiction require "
            "confirmation by a qualified radiologist before clinical use.",
            "No surgical or interventional management is recommended by this "
            "automated output.",
        ]

        # Impression (max configured points, conservative).
        imp = []
        sup = [
            f for f in case.findings
            if f.finding_type == FindingType.DISC_SPACE_NARROWING
            or f.evidence_status == EvidenceStatus.SUPPORTED
        ]
        for f in sup[:2]:
            imp.append(f.final_language)
        if case.low_confidence_findings:
            imp.append(
                "Automated multilevel findings include low-confidence "
                "predictions that require radiologist confirmation."
            )
        if contradicted_findings:
            imp.append(
                "Automated predictions at some levels are not corroborated by "
                "independent geometric analysis and should not be interpreted "
                "as confirmed findings."
            )
        for _ in range(max(0, self.config.reporting.max_impression_points)):
            if len(imp) < 1:
                imp.append(
                    "No objectively supported structural findings were "
                    "identified by the automated pipeline; review is advised."
                )
                break
        report.impression = imp[: self.config.reporting.max_impression_points]

        # Traceability: every statement -> evidence ids (from findings).
        for f in case.findings:
            if f.final_language:
                report.traceability.append(TraceStatement(
                    statement=f.final_language,
                    evidence_ids=list(f.evidence_ids),
                ))
        for c in case.contradictions:
            report.traceability.append(TraceStatement(
                statement=c.description,
                evidence_ids=[c.contradiction_id],
            ))
        report.traceability.append(TraceStatement(
            statement=report.longitudinal_outlook,
            evidence_ids=[case.longitudinal_risk.status.value],
        ))
        report.traceability.append(TraceStatement(
            statement=DISCLAIMER,
            evidence_ids=["POLICY"],
        ))
        return report


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_report(data) -> Optional[DecisionSupportReport]:
    if not isinstance(data, dict):
        return None
    try:
        return DecisionSupportReport(**data)
    except Exception:  # noqa: BLE001
        # tolerate missing traceability (built after) and coerce types
        try:
            cleaned = _coerce_report_fields(data)
            return DecisionSupportReport(**cleaned)
        except Exception:  # noqa: BLE001
            return None


def _coerce_report_fields(data: dict) -> dict:
    out = dict(data)
    for key in ("objective_imaging_findings", "model_confidence",
                "contradictions", "provisional_findings", "impression",
                "confidence_limitations", "recommendations"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = [value]
        elif value is None:
            out[key] = []
    for key in ("overall_status", "clinical_correlation",
                "longitudinal_outlook", "disclaimer"):
        if not isinstance(out.get(key), str):
            out[key] = str(out.get(key) or "")
    if not out.get("disclaimer"):
        out["disclaimer"] = DISCLAIMER
    return out


def _case_json(case: StructuredCase) -> str:
    payload = {
        "case_id": case.case_id,
        "image_quality": case.image_quality.model_dump(),
        "levels": {
            lvl: {
                "ddd": _compact_finding(ls.ddd),
                "spondylolisthesis": _compact_finding(ls.spondylolisthesis),
                "disc_space_narrowing": _compact_finding(ls.disc_space_narrowing),
            }
            for lvl, ls in case.levels.items()
        },
        "findings": [_compact_finding(f) for f in case.findings],
        "contradictions": [c.model_dump() for c in case.contradictions],
        "clinical_context": {
            "age": case.clinical_context.age,
            "sex": case.clinical_context.sex,
            "pain_score": case.clinical_context.pain_score,
            "pain_duration_years": case.clinical_context.pain_duration_years,
            "symptom_correlations": case.clinical_context.symptom_correlations,
        },
        "longitudinal_risk": case.longitudinal_risk.model_dump(),
        "overall_status": case.overall_status.value,
        "requires_radiologist_review": case.requires_radiologist_review,
    }
    return json.dumps(payload, indent=1, default=str)


def _compact_finding(f) -> dict:
    if f is None:
        return None
    return {
        "finding_id": f.finding_id,
        "finding_type": f.finding_type.value,
        "level": f.level,
        "model_prediction": None if f.model_prediction is None else {
            "prediction_value": f.model_prediction.prediction_value,
            "prediction_label": f.model_prediction.prediction_label,
            "grading_system": f.model_prediction.grading_system,
            "raw_confidence": f.model_prediction.raw_confidence,
            "calibrated_probability": f.model_prediction.calibrated_probability,
            "confidence_status": (
                None if f.model_prediction.confidence_status is None
                else f.model_prediction.confidence_status.value),
        },
        "geometric_validation": None if f.geometric_validation is None else {
            "supports": f.geometric_validation.supports,
            "contradicts": f.geometric_validation.contradicts,
            "measurements": f.geometric_validation.measurements,
        },
        "raw_confidence": f.raw_confidence,
        "calibrated_probability": f.calibrated_probability,
        "confidence_status": f.confidence_status.value,
        "evidence_status": f.evidence_status.value,
        "evidence_score": f.evidence_score,
        "score_is_probability": f.score_is_probability,
        "contradiction": f.contradiction,
        "contradiction_severity": f.contradiction_severity.value,
        "require_review": f.require_review,
        "final_language": f.final_language,
        "evidence_ids": f.evidence_ids,
    }


def _clinical_sentence(case: StructuredCase) -> str:
    if case.clinical_context.symptom_correlations:
        return " ".join(case.clinical_context.symptom_correlations)
    return (
        "Clinical correlation should be performed by the reviewing clinician."
    )


def _limitations(case: StructuredCase) -> list[str]:
    limitations = []
    if case.image_quality.status.value != "adequate":
        limitations.append(
            f"Image quality was assessed as {case.image_quality.status.value}: "
            + ("; ".join(case.image_quality.issues) if case.image_quality.issues
               else "review required.")
        )
    if case.low_confidence_findings:
        limitations.append(
            f"{len(case.low_confidence_findings)} low-confidence automated "
            "finding(s) were not independently confirmed."
        )
    if case.contradictions:
        limitations.append(
            f"{len(case.contradictions)} contradiction(s) between model "
            "predictions and independent geometric validation remain unresolved."
        )
    if not case.longitudinal_risk.available:
        limitations.append(
            "No validated numeric longitudinal model was available; numeric "
            "progression risk is not reported."
        )
    limitations.append(
        "Automated geometric thresholds and model probabilities are not "
        "clinically validated surrogates for a radiologist's assessment."
    )
    return limitations


def _sorted_findings(case: StructuredCase):
    order = {FindingType.DISC_SPACE_NARROWING: 0,
             FindingType.DDD: 1,
             FindingType.SPONDYLOLISTHESIS: 2}
    return sorted(case.findings, key=lambda f: (order.get(f.finding_type, 9),
                                                f.level))


def generate_report(
    case: StructuredCase,
    config: Optional[PipelineConfig] = None,
    provider: str | None = None,
) -> DecisionSupportResult:
    return ReportGenerator(config, provider).generate(case)