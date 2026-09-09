"""
evidence_fusion.py
==================

Evidence fusion / scoring (spec §16).

The pipeline produces a deterministic evidence matrix per finding:

    MODEL_SUPPORT           GEOMETRIC_SUPPORT        CLINICAL_SUPPORT
    MODEL_CONTRADICTION     GEOMETRIC_CONTRADICTION  CONFIDENCE

These are NOT summed naively — they are weighted contributions from
different evidence types (weights are configurable in config.yaml).
The resulting score is an *evidence score*, never a medical probability.

Score -> qualitative evidence_status:

    STRONGLY_SUPPORTED   SUPPORTED   PROVISIONAL
    INDETERMINATE        CONTESTED   NOT_SUPPORTED

plus pipeline-specific statuses:
    LOW_CONFIDENCE_UNSUPPORTED
    LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT
    CONTRADICTED

Disclosure (spec §16): we never claim the fused score is a medical
probability unless the calibration pipeline has been clinically validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline_config import PipelineConfig
from evidence_schemas import (
    ClinicalSupport,
    ConfidenceStatus,
    EvidenceStatus,
    GeometricValidation,
    ModelPrediction,
)
from contradiction_engine import ContradictionResult, ContradictionSeverity


@dataclass
class EvidenceMatrix:
    """Signed, per-dimension evidence contributions for one finding."""
    model_support: float = 0.0
    geometric_support: float = 0.0
    clinical_support: float = 0.0
    model_contradiction: float = 0.0
    geometric_contradiction: float = 0.0
    confidence: float = 0.0          # multiplier 0..1
    score: float = 0.0               # weighted combination
    score_is_probability: bool = False


class EvidenceFuser:
    """Computes the evidence matrix + final evidence status per finding."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.get()
        self.w = self.config.evidence_weights
        self.t = self.config.evidence_score

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def build_matrix(
        self,
        model: Optional[ModelPrediction],
        geo: Optional[GeometricValidation],
        clinical: Optional[ClinicalSupport],
        contradiction: Optional[ContradictionResult],
    ) -> EvidenceMatrix:
        m = EvidenceMatrix()

        # --- MODEL_SUPPORT ---
        if model is not None and model.prediction_value is not None:
            base = abs(float(model.prediction_value)) / self._max_claim(model)
            self._clamped(base)
            m.model_support = base

        # --- GEOMETRIC_SUPPORT / CONTRADICTION ---
        if geo is not None and geo.supports is not None:
            if geo.supports:
                m.geometric_support = 1.0
            else:
                m.geometric_contradiction = 1.0

        # --- CLINICAL_SUPPORT ---
        if clinical is not None:
            if clinical.supports:
                m.clinical_support = 1.0
            elif clinical.contradicts:
                m.model_contradiction += 0.5

        # --- MODEL_CONTRADICTION from the deterministic engine ---
        if contradiction is not None and contradiction.contradiction:
            if contradiction.severity == ContradictionSeverity.MAJOR_CONTRADICTION:
                m.geometric_contradiction = max(m.geometric_contradiction, 1.0)
                m.model_contradiction = max(m.model_contradiction, 1.0)
            elif contradiction.severity in (
                ContradictionSeverity.MODERATE_CONTRADICTION,
                ContradictionSeverity.MINOR_CONTRADICTION,
            ):
                m.model_contradiction = max(m.model_contradiction, 0.75)

        # --- CONFIDENCE multiplier ---
        m.confidence = (
            float(model.raw_confidence)
            if model is not None and model.raw_confidence is not None
            else 0.0
        )

        return m

    @staticmethod
    def _max_claim(model: ModelPrediction) -> float:
        """Normalisation denominator for a claim magnitude."""
        ftype = str(model.finding_type)
        if ftype == "SPONDYLOLISTHESIS":
            return 100.0          # slip percentage
        if ftype == "DDD":
            return 5.0            # Pfirrmann grade 1..5
        return 1.0

    @staticmethod
    def _clamped(x: float) -> float:
        return max(0.0, min(1.0, x))

    # ------------------------------------------------------------------
    # Weighted score
    # ------------------------------------------------------------------

    def score(self, matrix: EvidenceMatrix) -> float:
        raw = (
            self.w.model_support * matrix.model_support
            + self.w.geometric_support * matrix.geometric_support
            + self.w.clinical_support * matrix.clinical_support
            + self.w.model_contradiction * matrix.model_contradiction
            + self.w.geometric_contradiction * matrix.geometric_contradiction
        )
        # Confidence down-weights everything when landmarks/model are uncertain.
        multiplier = self.w.confidence_multiplier + (1.0 - self.w.confidence_multiplier) * matrix.confidence
        return raw * max(0.0, multiplier)

    # ------------------------------------------------------------------
    # Final status resolution
    # ------------------------------------------------------------------

    def resolve_status(
        self,
        matrix: EvidenceMatrix,
        contradiction: Optional[ContradictionResult],
        model: Optional[ModelPrediction],
        geo: Optional[GeometricValidation],
    ) -> EvidenceStatus:
        """Return the deterministic final evidence status (spec §5 / §8)."""
        if model is None or model.prediction_value is None:
            return EvidenceStatus.NOT_SUPPORTED

        conf_status = model.confidence_status or ConfidenceStatus.LOW_CONFIDENCE
        low_conf = conf_status == ConfidenceStatus.LOW_CONFIDENCE
        geom_present = geo is not None and geo.supports is not None
        geom_positive = geo is not None and bool(geo.supports)

        # Hard rules first (deterministic, override the soft score).
        if contradiction is not None and contradiction.contradiction:
            # Explicitly-measured negative geometry directly falsifies the
            # model claim (spec §8 R1) -> CONTRADICTED at any severity.
            geom_explicit_negative = geo is not None and geo.supports is False
            if geom_explicit_negative:
                return EvidenceStatus.CONTRADICTED
            if contradiction.severity == ContradictionSeverity.MAJOR_CONTRADICTION:
                return EvidenceStatus.CONTRADICTED
            if contradiction.severity == ContradictionSeverity.MODERATE_CONTRADICTION:
                # severity/possible-translation mismatch -> indeterminate
                return EvidenceStatus.INDETERMINATE
            return EvidenceStatus.CONTESTED

        if contradiction is not None and contradiction.status_hint in (
            EvidenceStatus.NOT_CORROBORATED,
        ):
            return EvidenceStatus.NOT_CORROBORATED

        # DDD / generic: low-confidence handling (spec §5)
        if str(model.finding_type) in ("DDD", "DISC_SPACE_NARROWING"):
            if low_conf:
                if geom_positive:
                    return EvidenceStatus.LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT
                return EvidenceStatus.LOW_CONFIDENCE_UNSUPPORTED
            # not low confidence
            if geom_positive:
                return EvidenceStatus.SUPPORTED
            if not geom_present:
                return EvidenceStatus.PROVISIONAL
            return EvidenceStatus.PROVISIONAL

        # Spondylolisthesis with moderate/high confidence and no
        # contradiction -> base status on geometry presence.
        if geom_positive:
            return EvidenceStatus.SUPPORTED
        if not geom_present:
            return EvidenceStatus.NOT_CORROBORATED
        if low_conf:
            return EvidenceStatus.CONTESTED
        return EvidenceStatus.CONTESTED

    def categorical_score_status(self, score: float) -> EvidenceStatus:
        """Map a raw evidence score onto the qualitative bands."""
        if score >= self.t.strongly_supported:
            return EvidenceStatus.SUPPORTED
        if score >= self.t.supported:
            return EvidenceStatus.SUPPORTED
        if score >= self.t.provisional:
            return EvidenceStatus.PROVISIONAL
        if score >= self.t.indeterminate:
            return EvidenceStatus.INDETERMINATE
        return EvidenceStatus.NOT_SUPPORTED

    def fuzz(self, matrix: EvidenceMatrix) -> EvidenceMatrix:
        matrix.score = self.score(matrix)
        matrix.score_is_probability = False   # never a medical probability
        return matrix