"""
schemas.py
==========

Structured data contracts shared across every agent in the pipeline.

Using pydantic keeps fields typed and lets the LLM agents return clean,
validatable JSON (see `SchemaModel` helpers for parsing LLM output).

Key objects
-----------
- PatientInput      : raw case inputs (image path + clinical + symptoms text)
- StructuredSymptoms: free-text symptoms parsed into structured fields
- LevelFinding      : per-disc-level numeric finding for ONE disease
- EvidenceCard      : full machine-output evidence (what the models say)
- ClinicalInterpretation : fusion agent's reasoned assessment
- VerificationResult: critique agent's audit of the interpretation
- LongitudinalAssessment: progression prediction + agent explanation
- FinalReport       : radiologist-style narrative + tables
- PipelineResult    : everything bundled for the UI / export
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from config import DISC_LEVELS


# ------------------------------------------------------------------
# Structured schemas returned by LLM agents (pydantic)
# ------------------------------------------------------------------

class StructuredSymptoms(BaseModel):
    """Free-text symptoms -> structured clinical signs."""
    pain_location: str = Field(default="", description="e.g. lower back (lumbar)")
    pain_radiation: str = Field(
        default="", description="radiation, e.g. to legs/buttock/bilateral"
    )
    neuro_symptoms: str = Field(
        default="", description="numbness, tingling, weakness, claudication"
    )
    pain_quality: str = Field(
        default="", description="sharp/dull/burning/radicular/mechanical"
    )
    exacerbating: str = Field(default="", description="what worsens the pain")
    relieving: str = Field(default="", description="what relieves the pain")
    pain_duration_hint: str = Field(
        default="", description="how long the pain has been present"
    )
    red_flags: str = Field(
        default="", description="any red-flag symptoms (cauda equina, fever, etc.)"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="agent confidence the extraction matches the patient's words",
    )


class LevelFinding(BaseModel):
    """Numeric finding for a single disc level for one disease."""
    level: str
    grade: Optional[float] = None          # DDD grade 0-4 (or None)
    slip_percent: Optional[float] = None   # spondylolisthesis slip % (or None)
    meyerding: str = ""                    # Meyerding grade label
    severity: str = ""                     # DDD severity label
    raw_confidence: float = 0.0            # model's raw confidence 0-1
    calibrated_probability: float = 0.0    # calibrated P(disease) 0-1
    localization_quality: float = 0.0      # landmark quality 0-1
    evidence: str = ""                     # human-readable evidence for the finding


class EvidenceCard(BaseModel):
    """All machine-produced evidence for a case (vision + longitudinal)."""
    image_processed: bool = False
    image_name: str = ""
    landmark_points: list[list[float]] = Field(default_factory=list)
    landmark_conf: list[float] = Field(default_factory=list)
    geometric_indicators: list[dict] = Field(default_factory=list)
    ddd: list[LevelFinding] = Field(default_factory=list)
    spondy: list[LevelFinding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Differential(BaseModel):
    """A single candidate diagnosis with evidence ties."""
    disease: str
    likelihood: float = Field(ge=0.0, le=1.0, description="0-1 probability")
    rationale: str = Field(default="", description="why, with cited evidence")
    evidence_ids: list[str] = Field(default_factory=list, description="evidence citations")


class ClinicalInterpretation(BaseModel):
    """Fusion agent's reasoned interpretation (per disease + overall)."""
    summaries: dict[str, str] = Field(
        default_factory=dict, description="per-disease narrative summary"
    )
    differentials: list[Differential] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class VerificationItem(BaseModel):
    """Critique of one diagnosis/claim from the fusion agent."""
    claim: str = ""
    supports: str = ""            # evidence supporting the claim
    conflicts: str = ""           # evidence contradicting it (if any)
    status: str = "confirmed"     # confirmed | contested | rejected | low_confidence
    correction: str = ""          # suggested correction if contested/rejected
    agree_probability: float = Field(default=1.0, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Critique agent's audit of the clinical interpretation."""
    overall_status: str = "approved"   # approved | needs_revision | rejected
    items: list[VerificationItem] = Field(default_factory=list)
    summary: str = ""
    consistency_notes: list[str] = Field(default_factory=list)


class LongitudinalAssessment(BaseModel):
    """Progression prediction from the longitudinal model + agent explanation."""
    years_ahead: float = 5.0
    overall_risk: float = 0.0
    per_level_future_grade: list[float] = Field(default_factory=list)
    per_level_worsening_risk: list[float] = Field(default_factory=list)
    narrative: str = ""


class FinalReport(BaseModel):
    """Radiologist-style written report generated by the report-writer agent."""
    impression: str = ""
    findings: str = ""
    diagnoses: list[dict] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    confidence_statement: str = ""
    caveats: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Everything the pipeline produced for one case (UI / export)."""
    patient: dict = Field(default_factory=dict)
    symptoms: Optional[StructuredSymptoms] = None
    evidence: Optional[EvidenceCard] = None
    interpretation: Optional[ClinicalInterpretation] = None
    verification: Optional[VerificationResult] = None
    longitudinal: Optional[LongitudinalAssessment] = None
    report: Optional[FinalReport] = None
    annotated_image_path: str = ""
    model_status: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        return self.model_dump()


# ------------------------------------------------------------------
# Plain (non-pydantic) input/output objects
# ------------------------------------------------------------------

@dataclass
class PatientInput:
    """Raw inputs for one case."""
    image_path: str = ""
    age: Optional[int] = None
    sex: str = ""
    pain_scale: Optional[float] = None     # VAS 0-10
    modality: str = ""                     # xray | mri
    pain_years: Optional[float] = None
    start_year: Optional[int] = None
    symptoms_text: str = ""
    years_ahead: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Helpers to build per-level findings
# ------------------------------------------------------------------

def _level_map(items: list[LevelFinding]):
    return {f.level: f for f in items}


def build_evidence_card(
    points,
    localization_conf,
    pixels,
    geo,
    ddd_grade,
    ddd_conf,
    spondy_slip_pct,
    spondy_conf,
    image_name: str = "",
    calibrate=None,
) -> EvidenceCard:
    """Assemble a full EvidenceCard from raw vision-engine arrays."""
    card = EvidenceCard(image_processed=True, image_name=image_name)
    card.landmark_points = [[float(x), float(y)] for x, y in points]
    card.landmark_conf = [float(c) for c in localization_conf]
    card.geometric_indicators = list(geo)

    for i, level in enumerate(DISC_LEVELS):
        # ---- DDD finding ----
        d_grade = float(ddd_grade[i])
        d_rc = float(ddd_conf[i])
        d_sev, d_prob = calibrate_ddd(d_grade, d_rc) if calibrate else (
            severity_grade(d_grade), d_rc
        )
        d_ev = (
            f"Model DDD grade {d_grade:.2f}/4 ({d_sev}); "
            f"calibrated probability {d_prob:.2f}; model confidence {d_rc:.2f}."
        )
        ddd = LevelFinding(
            level=level, grade=round(d_grade, 3), severity=d_sev,
            raw_confidence=round(d_rc, 3),
            calibrated_probability=round(d_prob, 3),
            localization_quality=round(float(localization_conf[i]), 3),
            evidence=d_ev,
        )

        # ---- Spondylolisthesis finding ----
        s_slip = float(spondy_slip_pct[i])
        s_rc = float(spondy_conf[i])
        s_grade, s_prob = calibrate_spondy(s_slip, s_rc) if calibrate else (
            meyer_grade(s_slip), s_rc
        )
        s_ev = (
            f"Model slip {s_slip:.1f}% ({s_grade}); "
            f"calibrated probability {s_prob:.2f}; model confidence {s_rc:.2f}."
        )
        spondy = LevelFinding(
            level=level, slip_percent=round(s_slip, 1), meyerding=s_grade,
            raw_confidence=round(s_rc, 3),
            calibrated_probability=round(s_prob, 3),
            localization_quality=round(float(localization_conf[i]), 3),
            evidence=s_ev,
        )

        card.ddd.append(ddd)
        card.spondy.append(spondy)

    return card


def severity_grade(grade: float) -> str:
    if grade < 0.75:
        return "None"
    if grade < 1.75:
        return "Mild"
    if grade < 2.75:
        return "Moderate"
    return "Severe"


def meyer_grade(slip_percent: float) -> str:
    if slip_percent < 5.0:
        return "Normal (<5%)"
    if slip_percent < 25.0:
        return "Grade I"
    if slip_percent < 50.0:
        return "Grade II"
    if slip_percent < 75.0:
        return "Grade III"
    if slip_percent <= 100.0:
        return "Grade IV"
    return "Grade V (spondyloptosis)"


def calibrate_ddd(grade: float, raw_conf: float) -> tuple[str, float]:
    """
    Deterministic fallback calibration (no fitted calibrator):
    P(DDD) blends the raw confidence with how far the grade deviates
    from normal. When a fitted calibrator exists this is overridden in
    `calibration.py`.
    """
    severity = severity_grade(grade)
    severity_prior = {"None": 0.10, "Mild": 0.45, "Moderate": 0.75,
                      "Severe": 0.90}.get(severity, 0.5)
    probability = 0.6 * severity_prior + 0.4 * raw_conf
    return severity, max(0.0, min(1.0, probability))


def calibrate_spondy(slip_percent: float, raw_conf: float) -> tuple[str, float]:
    slip = meyer_grade(slip_percent)
    if slip == "Normal (<5%)":
        prior = 0.10
    else:
        prior = min(0.95, 0.35 + float(slip_percent) / 100.0)
    probability = 0.6 * prior + 0.4 * raw_conf
    return slip, max(0.0, min(1.0, probability))
