"""
contradiction_engine.py
=======================

Deterministic, rule-based contradiction detection (spec §8 and §9).

The contradiction engine runs BEFORE any LLM generation. It compares:

    - the model prediction  (LEVEL 3 evidence)
    - independent geometric measurements (LEVEL 2 evidence)
    - optionally clinical context (LEVEL 4)

and produces a structured Contradiction record with a severity:

    NO_CONTRADICTION / MINOR / MODERATE / MAJOR

Key rules (spec §8, §29):

  R1  spondy model positive  + geometry explicit negative
      -> contradiction, severity by claim strength, status CONTRADICTED
  R2  spondy model positive  + geometry positive but severity unvalidated
      -> MODERATE contradiction, status INDETERMINATE / PROVISIONAL
  R3  spondy model positive  + geometry ABSENT
      -> NOT_CORROBORATED (no contradiction, but nothing supports it)
  R4  model low confidence   + no geometric support
      -> LOW_CONFIDENCE_UNSUPPORTED
  R5  model low confidence   + geometric support present
      -> LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT
  R6  model positive  + geometry positive + compatible clinical
      -> SUPPORTED

Severity mapping (spec §9):
  - Grade III claim + explicit negative geometry            -> MAJOR
  - Grade III claim + "possible translation" only           -> MODERATE
  - Grade III claim + no geometry at all                    -> MODERATE
                                                              (severity unvalidated)
The exact thresholds are configurable (config.yaml).
"""

from __future__ import annotations

from typing import Optional

from pipeline_config import PipelineConfig
from evidence_schemas import (
    ClinicalSupport,
    Contradiction,
    ContradictionSeverity,
    EvidenceStatus,
    Finding,
    FindingType,
    GeometricValidation,
    ModelPrediction,
)


class ContradictionResult:
    """Deterministic output for one finding."""
    def __init__(
        self,
        contradiction: bool,
        severity: ContradictionSeverity,
        reason: str,
        rule_id: str,
        status_hint: Optional[EvidenceStatus] = None,
        review_required: bool = True,
        cont_geometric_claim: str = "",
    ):
        self.contradiction = contradiction
        self.severity = severity
        self.reason = reason
        self.rule_id = rule_id
        self.status_hint = status_hint
        self.review_required = review_required
        self.geometric_claim = cont_geometric_claim


class ContradictionEngine:
    """Evaluates one finding against LEVEL-2 geometric evidence."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.get()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Optional[ModelPrediction],
        geo: Optional[GeometricValidation],
        clinical: Optional[ClinicalSupport],
    ) -> ContradictionResult:
        """Run the rule set for a single finding."""
        if model is None or model.prediction_value is None:
            # No model prediction -> nothing to contradict.
            return ContradictionResult(
                contradiction=False,
                severity=ContradictionSeverity.NO_CONTRADICTION,
                reason="No model prediction for this level/finding.",
                rule_id="R0",
                status_hint=EvidenceStatus.NOT_SUPPORTED,
                review_required=False,
            )

        ftype = FindingType(model.finding_type)

        if ftype == FindingType.SPONDYLOLISTHESIS:
            return self._evaluate_spondy(model, geo, clinical)
        if ftype in (FindingType.DDD, FindingType.DISC_SPACE_NARROWING):
            return self._evaluate_ddd(model, geo, clinical)
        return ContradictionResult(
            contradiction=False,
            severity=ContradictionSeverity.NO_CONTRADICTION,
            reason="No contradiction rules defined for this finding type.",
            rule_id="R0_X",
            review_required=False,
        )

    # ------------------------------------------------------------------
    # Spondylolisthesis rules
    # ------------------------------------------------------------------

    def _evaluate_spondy(self, model, geo, clinical) -> ContradictionResult:
        slip = float(model.prediction_value)
        severe = slip >= self.config.validation.severe_slip_percent
        claim_label = model.prediction_label or _slip_claim_label(slip, severe)

        if geo is None or geo.supports is None:
            # R3: geometry ABSENT -> cannot corroborate, not a contradiction.
            return ContradictionResult(
                contradiction=False,
                severity=ContradictionSeverity.NO_CONTRADICTION,
                reason=(
                    f"Spondylolisthesis model predicts {claim_label}; "
                    "independent geometric validation is unavailable for this "
                    "level, so the prediction cannot be corroborated."
                ),
                rule_id="R3",
                status_hint=EvidenceStatus.NOT_CORROBORATED,
                review_required=True,
            )

        if geo.supports is True:
            # Geometry positive ("listhesis_possible"), but geometry alone
            # cannot validate the SEVERITY (spec §6 / §9).
            if severe:
                return ContradictionResult(
                    contradiction=True,
                    severity=ContradictionSeverity.MODERATE_CONTRADICTION,
                    reason=(
                        f"Model predicts a severe slip ({claim_label}); geometric "
                        "analysis only confirms possible translation and does not "
                        "independently validate slip severity. Severity is "
                        "indeterminate pending specialist review."
                    ),
                    rule_id="R2",
                    status_hint=EvidenceStatus.INDETERMINATE,
                    review_required=True,
                    cont_geometric_claim="possible vertebral translation",
                )
            return ContradictionResult(
                contradiction=False,
                severity=ContradictionSeverity.NO_CONTRADICTION,
                reason="Model prediction is consistent with geometric analysis.",
                rule_id="R6",
                status_hint=EvidenceStatus.SUPPORTED,
                review_required=False,
                cont_geometric_claim="geometric translation detected",
            )

        # geometry explicit negative: geo.supports is False
        if severe:
            severity = ContradictionSeverity.MAJOR_CONTRADICTION
        else:
            severity = ContradictionSeverity.MODERATE_CONTRADICTION
        return ContradictionResult(
            contradiction=True,
            severity=severity,
            reason=(
                f"Model predicts {claim_label}; independent geometric "
                "measurements do not corroborate vertebral translation "
                "at this level. The prediction is not corroborated by "
                "geometric analysis."
            ),
            rule_id="R1",
            status_hint=EvidenceStatus.CONTRADICTED,
            review_required=True,
            cont_geometric_claim="no vertebral translation detected",
        )

    # ------------------------------------------------------------------
    # DDD rules
    # ------------------------------------------------------------------

    def _evaluate_ddd(self, model, geo, clinical) -> ContradictionResult:
        narrowed = geo is not None and bool(geo.supports)
        geometric_present = geo is not None and geo.supports is not None

        if not narrowed:
            return ContradictionResult(
                contradiction=False,
                severity=ContradictionSeverity.NO_CONTRADICTION,
                reason=(
                    "No independent geometric support for disc-space narrowing "
                    "at this level."
                    + ("" if geometric_present else " (geometry unavailable)")
                ),
                rule_id="R4",
                status_hint=None,     # resolved by confidence in fusion
                review_required=False,
            )

        return ContradictionResult(
            contradiction=False,
            severity=ContradictionSeverity.NO_CONTRADICTION,
            reason="Automated DDD prediction supported by independent " \
                   "geometric disc-space measurement.",
            rule_id="R5",
            status_hint=None,          # resolved by confidence in fusion
            review_required=False,
        )


def _slip_claim_label(slip: float, severe: bool) -> str:
    label = f"slip of {slip:.1f}%"
    return label


def build_contradiction(
    contradiction_id: str,
    finding_id: str,
    ftype: FindingType,
    level: str,
    result: ContradictionResult,
    model_claim: str,
) -> Contradiction:
    """Pack a ContradictionResult into the persisted Contradiction model."""
    return Contradiction(
        contradiction_id=contradiction_id,
        finding_id=finding_id,
        finding_type=ftype,
        level=level,
        model_claim=model_claim,
        geometric_claim=result.geometric_claim,
        severity=result.severity,
        description=result.reason,
        review_required=result.review_required,
    )