"""
evidence_schemas.py
===================

Typed data contracts for the deterministic evidence/validation safety
layer (confidence status, geometric validation, contradiction engine,
evidence fusion, structured intermediate JSON, conservative report).

These models deliberately keep raw model outputs, calibrated outputs,
and geometric measurements in separate fields — the pipeline MUST never
conflate them (spec §3).

Evidence provenance
-------------------
Every statement in the final report is traceable to evidence entries.
Each evidence entry carries a stable `evidence_id` (E001, E002, ...),
a `source` tag and the exact numeric values it contributes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------

class ConfidenceStatus(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MODERATE_CONFIDENCE = "MODERATE_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class EvidenceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    SUPPORTED = "SUPPORTED"
    PROBABLE = "PROBABLE"
    PROVISIONAL = "PROVISIONAL"
    INDETERMINATE = "INDETERMINATE"
    CONTESTED = "CONTESTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_CONFIDENCE_UNSUPPORTED = "LOW_CONFIDENCE_UNSUPPORTED"
    LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT = (
        "LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT"
    )
    NOT_CORROBORATED = "NOT_CORROBORATED"
    CONTRADICTED = "CONTRADICTED"


class ContradictionSeverity(str, Enum):
    NO_CONTRADICTION = "NO_CONTRADICTION"
    MINOR_CONTRADICTION = "MINOR_CONTRADICTION"
    MODERATE_CONTRADICTION = "MODERATE_CONTRADICTION"
    MAJOR_CONTRADICTION = "MAJOR_CONTRADICTION"


class OverallStatus(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    INADEQUATE_INPUT = "INADEQUATE_INPUT"


class ImageQualityStatus(str, Enum):
    ADEQUATE = "adequate"
    LIMITED = "limited"
    INADEQUATE = "inadequate"


class FindingType(str, Enum):
    DDD = "DDD"
    SPONDYLOLISTHESIS = "SPONDYLOLISTHESIS"
    DISC_SPACE_NARROWING = "DISC_SPACE_NARROWING"
    OTHER = "OTHER"


class RiskStatus(str, Enum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    AVAILABLE = "AVAILABLE"


# ------------------------------------------------------------------
# Evidence provenance
# ------------------------------------------------------------------

class EvidenceSource(str, Enum):
    IMAGE_QUALITY = "image_quality"
    LEVEL_VALIDATION = "level_validation"
    GEOMETRY = "geometry"
    DDD_MODEL = "DDD_model"
    SPONDYLOLISTHESIS_MODEL = "spondylolisthesis_model"
    CLINICAL = "clinical"
    LONGITUDINAL_MODEL = "longitudinal_model"


class EvidenceEntry(BaseModel):
    """One auditable evidence item (stable ID for traceability)."""
    evidence_id: str = ""
    source: EvidenceSource
    level: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    note: str = ""
    calibrated: bool = False       # does `values` hold a calibrated probability?
    measurement: bool = False      # independent geometric/anatomical measurement


# ------------------------------------------------------------------
# Per-finding objects
# ------------------------------------------------------------------

class ModelPrediction(BaseModel):
    """Raw + calibrated model output for ONE candidate finding."""
    finding_type: str
    level: str
    # Disease-specific value (Pfirrmann grade, slip %, ...). `None` when
    # the model produced no prediction.
    prediction_value: Optional[float] = None
    prediction_label: str = ""
    grading_system: str = ""          # e.g. "Pfirrmann I-V", "Meyerding grade"
    raw_confidence: Optional[float] = None
    calibrated_probability: Optional[float] = None
    confidence_status: Optional[ConfidenceStatus] = None
    model_version: str = ""


class GeometricValidation(BaseModel):
    """Independent geometric/anatomical evidence for a level.

    `supports` is a tri-state:
      - True  -> geometry is POSITIVE (e.g. listhesis_possible=True)
      - False -> geometry is NEGATIVE and directly measured (contradicts)
      - None  -> geometry is ABSENT/UNAVAILABLE (no contradiction, no support)
    """
    source: str = "geometry"
    level: str
    measurements: dict[str, Any] = Field(default_factory=dict)
    supports: Optional[bool] = None
    contradicts: Optional[bool] = None
    measurement_quality: Optional[float] = None
    landmark_confidence: Optional[float] = None
    note: str = ""


class ClinicalSupport(BaseModel):
    """Clinical context — kept strictly separate from imaging findings."""
    source: str = "clinical"
    compatible: Optional[bool] = None
    supports: Optional[bool] = None
    contradicts: Optional[bool] = None
    note: str = ""


class Finding(BaseModel):
    """A single candidate finding with its full evidence determination."""
    finding_id: str
    finding_type: FindingType
    level: str
    model_prediction: Optional[ModelPrediction] = None
    geometric_validation: Optional[GeometricValidation] = None
    clinical_support: Optional[ClinicalSupport] = None

    raw_confidence: Optional[float] = None
    calibrated_probability: Optional[float] = None
    confidence_status: ConfidenceStatus = ConfidenceStatus.LOW_CONFIDENCE

    evidence_status: EvidenceStatus = EvidenceStatus.INDETERMINATE
    evidence_score: float = 0.0
    evidence_score_type: str = ""     # e.g. "weighted_matrix"
    score_is_probability: bool = False  # NEVER True for unvalidated fusion

    contradiction: bool = False
    contradiction_severity: ContradictionSeverity = ContradictionSeverity.NO_CONTRADICTION
    contradiction_reason: str = ""

    require_review: bool = True
    final_language: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A deterministic contradiction record (model vs other evidence)."""
    contradiction_id: str
    finding_id: str
    finding_type: FindingType
    level: str
    model_claim: str
    geometric_claim: str = ""
    severity: ContradictionSeverity
    description: str
    review_required: bool = True


# ------------------------------------------------------------------
# Structured intermediate representation (spec §15)
# ------------------------------------------------------------------

class ImageQuality(BaseModel):
    status: ImageQualityStatus = ImageQualityStatus.ADEQUATE
    confidence: Optional[float] = None
    issues: list[str] = Field(default_factory=list)
    evidence_id: str = ""


class LevelStructure(BaseModel):
    """Per-level container storing BOTH DDD and spondy findings."""
    level: str
    ddd: Optional[Finding] = None
    spondylolisthesis: Optional[Finding] = None
    disc_space_narrowing: Optional[Finding] = None


class ClinicalContext(BaseModel):
    age: Optional[int] = None
    sex: str = ""
    pain_score: Optional[float] = None
    pain_duration_years: Optional[float] = None
    symptoms: dict[str, Any] = Field(default_factory=dict)
    symptom_correlations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LongitudinalRiskResult(BaseModel):
    available: bool = False
    horizon_years: int = 5
    score: Optional[float] = None        # MUST stay None when unavailable
    status: RiskStatus = RiskStatus.NOT_AVAILABLE
    per_level_risk: list[float] = Field(default_factory=list)
    note: str = ""


class StructuredCase(BaseModel):
    """The validated structured evidence LLM is allowed to see."""
    case_id: str
    pipeline_version: str = ""
    image_quality: ImageQuality = Field(default_factory=ImageQuality)
    levels: dict[str, LevelStructure] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    clinical_context: ClinicalContext = Field(default_factory=ClinicalContext)
    longitudinal_risk: LongitudinalRiskResult = Field(
        default_factory=LongitudinalRiskResult)
    overall_status: OverallStatus = OverallStatus.NEEDS_REVIEW
    requires_radiologist_review: bool = True
    evidence_registry: dict[str, EvidenceEntry] = Field(default_factory=dict)
    low_confidence_findings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def findings_by_type(self, t: FindingType) -> list[Finding]:
        return [f for f in self.findings if f.finding_type == t]


# ------------------------------------------------------------------
# Final report (LLM output, schema-validated)
# ------------------------------------------------------------------

class TraceStatement(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class DecisionSupportReport(BaseModel):
    """Conservative decision-support report (section 20 structure)."""
    overall_status: str = ""
    objective_imaging_findings: list[str] = Field(default_factory=list)
    model_confidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    clinical_correlation: str = ""
    provisional_findings: list[str] = Field(default_factory=list)
    longitudinal_outlook: str = ""
    impression: list[str] = Field(default_factory=list)
    confidence_limitations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    disclaimer: str = ""
    traceability: list[TraceStatement] = Field(default_factory=list)

    @property
    def text(self) -> str:
        sections = []
        if self.overall_status:
            sections.append(f"## Overall Status\n{self.overall_status}")
        if self.objective_imaging_findings:
            sections.append("## Objective Imaging Findings\n" + "\n".join(
                f"{i}. {f}" for i, f in enumerate(
                    self.objective_imaging_findings, start=1)))
        if self.model_confidence:
            sections.append("## Model Confidence\n" + "\n".join(
                f"- {x}" for x in self.model_confidence))
        if self.contradictions:
            sections.append("## Contradictions\n" + "\n".join(
                f"- {x}" for x in self.contradictions))
        if self.clinical_correlation:
            sections.append("## Clinical Correlation\n" + self.clinical_correlation)
        if self.provisional_findings:
            sections.append("## Provisional Findings\n" + "\n".join(
                f"- {x}" for x in self.provisional_findings))
        if self.longitudinal_outlook:
            sections.append("## Longitudinal Outlook\n" + self.longitudinal_outlook)
        if self.impression:
            sections.append("## Impression\n" + "\n".join(
                f"{i}. {p}" for i, p in enumerate(self.impression, start=1)))
        if self.confidence_limitations:
            sections.append("## Confidence / Limitations\n" + "\n".join(
                f"- {x}" for x in self.confidence_limitations))
        if self.recommendations:
            sections.append("## Recommendations\n" + "\n".join(
                f"- {x}" for x in self.recommendations))
        if self.disclaimer:
            sections.append("## Disclaimer\n" + self.disclaimer)
        return "\n\n".join(sections)


class DecisionSupportResult(BaseModel):
    """Everything the decision-support pipeline produced for one case."""
    case: StructuredCase
    report: Optional[DecisionSupportReport] = None
    report_validated: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    report_attempts: int = 0
    llm_used: bool = False
    audit_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()