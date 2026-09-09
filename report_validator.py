"""
report_validator.py
===================

Deterministic validation of the final decision-support report (spec §23).

Before the report is returned:

  - REJECT patterns are scanned against the report text. Any hit marks the
    report invalid (triggers regeneration).
  - REQUIRED content is checked given structured evidence conditions
    (low confidence present, contradictions present, longitudinal risk
    unavailable, radiologist disclaimer ...). A missing required item also
    marks the report invalid.
  - Structural constraints (impression length, traceability non-empty,
    numeric-risk-without-model) are verified.

The validator exists precisely so the LLM cannot override the structured
evidence (§25): claims the rules cannot support are rejected, not patched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from evidence_schemas import (
    DecisionSupportReport,
    EvidenceStatus,
    ImageQualityStatus,
    StructuredCase,
)
from pipeline_config import PipelineConfig


@dataclass
class ValidationResult:
    valid: bool = False
    errors: list[str] = field(default_factory=list)


class ReportValidator:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.get()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        case: StructuredCase,
        report: Optional[DecisionSupportReport],
    ) -> ValidationResult:
        errors: list[str] = []

        if report is None:
            errors.append("No report produced.")
            return ValidationResult(valid=False, errors=errors)

        text = report.text

        # --- structural validation ---
        self._check_structure(case, report, text, errors)

        # --- reject patterns ---
        self._check_reject_patterns(case, text, errors)

        # --- required content ---
        self._check_required_content(case, text, errors)

        # --- discipline rules ---
        if case.longitudinal_risk.score is None and not self._risk_text_allowed(case, report):
            errors.append(
                "Report contains a numeric longitudinal risk although no "
                "validated longitudinal model was available."
            )
        if self.config.reporting.require_radiologist_disclaimer:
            if not report.disclaimer or "radiologist" not in (
                report.disclaimer + text
            ).lower():
                errors.append("Radiologist review disclaimed missing.")
        if len(report.impression) > self.config.reporting.max_impression_points:
            errors.append(
                f"Impression has {len(report.impression)} points; allowed at "
                f"most {self.config.reporting.max_impression_points}."
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_structure(self, case, report, text, errors):
        inadequate = case.image_quality.status == ImageQualityStatus.INADEQUATE
        if not report.objective_imaging_findings and not inadequate:
            errors.append("Objective imaging findings section is empty.")
        if not report.overall_status:
            errors.append("Overall status is empty.")
        if not report.traceability:
            errors.append("No traceability mapping (statement -> evidence_ids).")
        else:
            empty = [t.statement for t in report.traceability if not t.evidence_ids]
            if empty:
                errors.append(
                    f"{len(empty)} traceable statement(s) lack evidence_ids."
                )

    def _check_reject_patterns(self, case, text, errors):
        for rule in self.config.validation.reject_patterns:
            pattern = rule.get("regex", "")
            reason = rule.get("reason", "unsupported claim")
            allowed = rule.get("allowed_when_causal_language", True)
            if not pattern:
                continue
            if not allowed and self.config.reporting.allow_causal_language:
                continue
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                errors.append(f"Reject pattern matched: {reason} ({pattern!r})")

    def _check_required_content(self, case, text, errors):
        probes = self._condition_probes(case)
        for rule in self.config.validation.required_content:
            condition = rule.get("condition", "always")
            pattern = rule.get("regex", "")
            reason = rule.get("reason", "missing content")
            if not pattern:
                continue
            if condition == "always" or probes.get(condition):
                if not re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    errors.append(f"Required content missing: {reason} ({pattern!r})")

    def _condition_probes(self, case: StructuredCase) -> dict[str, bool]:
        statuses = {f.evidence_status for f in case.findings}
        return {
            "any_low_confidence": bool(case.low_confidence_findings),
            "any_contradiction": bool(case.contradictions),
            "risk_available_false": not case.longitudinal_risk.available,
            "any_confirmed": EvidenceStatus.CONFIRMED in statuses,
        }

    def _risk_text_allowed(self, case, report) -> bool:
        """Numeric risk statements are only allowed when the model exists."""
        if case.longitudinal_risk.available:
            return True
        # Fallback: even with the model declared, never allow a literal zero
        # placeholder (spec §14 / §17).
        text = report.longitudinal_outlook
        return not re.search(
            r"(0\.00|0\s*(%|percent)|risk\s+of\s+0\b)", text,
            re.IGNORECASE,
        )