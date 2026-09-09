"""
structured_builder.py
=====================

Structured intermediate JSON generation (spec §15) + evidence fusion
orchestration for the decision-support pipeline.

Role
----
Takes raw, unvalidated inputs (DDD predictions, spondylolisthesis
predictions, independent geometric measurements, image quality, clinical
context) and produces:

    StructuredCase
        case_id
        image_quality
        levels: { "L1/L2": {ddd, spondylolisthesis, disc_space_narrowing}, ... }
        findings: [Finding, ...]          # each with confidence_status,
                                          # evidence_status, contradiction,
                                          # final_language, evidence_ids
        contradictions: [Contradiction, ...]
        clinical_context (objective imaging findings kept separate)
        longitudinal_risk (safe NOT_AVAILABLE handling)
        overall_status
        requires_radiologist_review
        evidence_registry (E001, E002, ... traceability)

Raw model outputs are preserved as-is (Level 3) while Level-2 geometric
measurements stay in their own fields — nothing is merged prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pipeline_config import PipelineConfig
from evidence_schemas import (
    ClinicalContext,
    ClinicalSupport,
    ConfidenceStatus,
    Contradiction,
    ContradictionSeverity,
    EvidenceEntry,
    EvidenceSource,
    EvidenceStatus,
    Finding,
    FindingType,
    GeometricValidation,
    ImageQuality,
    ImageQualityStatus,
    LevelStructure,
    LongitudinalRiskResult,
    ModelPrediction,
    OverallStatus,
    StructuredCase,
)
from confidence import classify_confidence, is_low_confidence
from contradiction_engine import ContradictionEngine, ContradictionResult, build_contradiction
from evidence_fusion import EvidenceFuser
from longitudinal_risk import evaluate_longitudinal_risk


@dataclass
class RawDDDPrediction:
    level: str
    pfirrmann_grade: Optional[float] = None
    raw_confidence: Optional[float] = None
    calibrated_probability: Optional[float] = None
    model_version: str = ""


@dataclass
class RawSpondyPrediction:
    level: str
    slip_percent: Optional[float] = None
    grade_label: str = ""
    raw_confidence: Optional[float] = None
    calibrated_probability: Optional[float] = None
    model_version: str = ""


@dataclass
class RawGeometry:
    level: str
    offset_ratio: Optional[float] = None
    listhesis_possible: Optional[bool] = None
    relative_space: Optional[float] = None
    space_narrowed: Optional[bool] = None
    landmark_confidence: Optional[float] = None
    measurement_quality: Optional[float] = None


@dataclass
class RawImageQuality:
    status: str = "adequate"          # adequate | limited | inadequate
    confidence: Optional[float] = None
    issues: list[str] = field(default_factory=list)


@dataclass
class RawCaseInput:
    """Plain (unvalidated) inputs for one case."""
    case_id: str = "case-unknown"
    image_quality: Optional[RawImageQuality] = None
    ddd_predictions: list[RawDDDPrediction] = field(default_factory=list)
    spondy_predictions: list[RawSpondyPrediction] = field(default_factory=list)
    geometry: list[RawGeometry] = field(default_factory=list)
    clinical_context: Optional[dict[str, Any]] = None
    symptoms: Optional[dict[str, Any]] = None
    longitudinal_horizon: int = 5


# ======================================================================
# Builder
# ======================================================================

class StructuredCaseBuilder:
    """Assembles a validated StructuredCase from raw inputs."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.get()
        self.contradictions_engine = ContradictionEngine(self.config)
        self.fuser = EvidenceFuser(self.config)
        self._evidence_counter = 0
        self._evidence_registry: dict[str, EvidenceEntry] = {}

    # ------------------------------------------------------------------
    # Evidence registry
    # ------------------------------------------------------------------

    def _register(self, source: EvidenceSource, *, level: str = "",
                  values: dict | None = None, note: str = "",
                  measurement: bool = False,
                  calibrated: bool = False) -> str:
        self._evidence_counter += 1
        eid = f"E{self._evidence_counter:03d}"
        self._evidence_registry[eid] = EvidenceEntry(
            evidence_id=eid, source=source, level=level,
            values=dict(values or {}), note=note,
            calibrated=calibrated, measurement=measurement,
        )
        return eid

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self, raw: RawCaseInput) -> StructuredCase:
        self._evidence_counter = 0
        self._evidence_registry = {}

        image_quality = self._build_image_quality(raw)

        levels: dict[str, LevelStructure] = {
            lvl: LevelStructure(level=lvl)
            for lvl in _KNOWN_LEVELS
        }
        findings: list[Finding] = []
        contradictions: list[Contradiction] = []

        # DDD findings
        for pred in raw.ddd_predictions:
            if pred.level not in levels:
                continue
            geo = self._find_geometry(raw.geometry, pred.level)
            f = self._build_ddd_finding(pred, geo, raw)
            if f is not None:
                levels[pred.level].ddd = f
                findings.append(f)

        # Spondylolisthesis findings
        for pred in raw.spondy_predictions:
            if pred.level not in levels:
                continue
            geo = self._find_geometry(raw.geometry, pred.level)
            f = self._build_spondy_finding(pred, geo, raw)
            if f is not None:
                levels[pred.level].spondylolisthesis = f
                findings.append(f)
                if f.contradiction:
                    contradictions.append(self._pack_contradiction(f))

        # Disc-space narrowing findings derived from geometry (Level 2).
        for g in raw.geometry:
            if g.level not in levels:
                continue
            if g.space_narrowed is True and g.relative_space is not None:
                f = self._build_narrowing_finding(g)
                levels[g.level].disc_space_narrowing = f
                findings.append(f)

        clinical = self._build_clinical_context(raw)

        longitudinal = evaluate_longitudinal_risk(
            horizon_years=raw.longitudinal_horizon, config=self.config
        )

        case = StructuredCase(
            case_id=raw.case_id,
            pipeline_version=self.config.pipeline.version,
            image_quality=image_quality,
            levels=levels,
            findings=findings,
            contradictions=contradictions,
            clinical_context=clinical,
            longitudinal_risk=longitudinal,
            evidence_registry=self._evidence_registry,
            requires_radiologist_review=True,
        )
        case.low_confidence_findings = [
            f.finding_id for f in findings
            if f.confidence_status == ConfidenceStatus.LOW_CONFIDENCE
        ]
        case.overall_status = self._overall_status(case)
        case.requires_radiologist_review = self._requires_review(case)
        return case

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_image_quality(self, raw: RawCaseInput) -> ImageQuality:
        status = raw.image_quality.status if raw.image_quality else "adequate"
        try:
            q = ImageQualityStatus(status)
        except ValueError:
            q = ImageQualityStatus.LIMITED
        confidence = raw.image_quality.confidence if raw.image_quality else None
        issues = list(raw.image_quality.issues) if raw.image_quality else []
        eid = self._register(
            EvidenceSource.IMAGE_QUALITY,
            values={"status": q.value, "confidence": confidence},
            note="Input image validity assessment.",
        )
        return ImageQuality(status=q, confidence=confidence,
                            issues=issues, evidence_id=eid)

    def _find_geometry(self, geos: list[RawGeometry], level: str) -> Optional[RawGeometry]:
        for g in geos:
            if g.level == level:
                return g
        return None

    # ------------------------------------------------------------------
    # DDD finding
    # ------------------------------------------------------------------

    def _build_ddd_finding(
        self,
        pred: RawDDDPrediction,
        geo: Optional[RawGeometry],
        raw: RawCaseInput,
    ) -> Optional[Finding]:
        if pred.pfirrmann_grade is None:
            return None

        finding_id = f"F_{pred.level}_DDD"
        conf_status = classify_confidence(pred.raw_confidence, self.config)

        geo_model = None
        if geo is not None:
            geo_values: dict[str, Any] = {}
            if geo.relative_space is not None:
                geo_values["relative_space"] = geo.relative_space
            if geo.space_narrowed is not None:
                geo_values["space_narrowed"] = geo.space_narrowed
            geo_eid = self._register(
                EvidenceSource.GEOMETRY, level=pred.level,
                values=geo_values,
                note="Independent disc-space geometric measurement.",
                measurement=True,
            )
            geo_model = GeometricValidation(
                level=pred.level,
                measurements=geo_values,
                supports=bool(geo.space_narrowed) if geo.space_narrowed is not None else None,
                contradicts=None,
                measurement_quality=geo.measurement_quality,
                landmark_confidence=geo.landmark_confidence,
            )

        model_eid = self._register(
            EvidenceSource.DDD_MODEL, level=pred.level,
            values={
                "pfirrmann_grade": pred.pfirrmann_grade,
                "raw_confidence": pred.raw_confidence,
                "calibrated_probability": pred.calibrated_probability,
            },
            note="Automated Pfirrmann grading prediction.",
        )

        model = ModelPrediction(
            finding_type="DDD",
            level=pred.level,
            prediction_value=pred.pfirrmann_grade,
            prediction_label=_pfirrmann_label(pred.pfirrmann_grade),
            grading_system="Pfirrmann I-V",
            raw_confidence=pred.raw_confidence,
            calibrated_probability=pred.calibrated_probability,
            confidence_status=conf_status,
            model_version=pred.model_version,
        )

        result = self.contradictions_engine.evaluate(model, geo_model, None)

        matrix = self.fuser.build_matrix(model, geo_model, None, result)
        matrix = self.fuser.fuzz(matrix)
        status = self.fuser.resolve_status(matrix, result, model, geo_model)
        score = matrix.score

        clinical_support = None
        f = Finding(
            finding_id=finding_id,
            finding_type=FindingType.DDD,
            level=pred.level,
            model_prediction=model,
            geometric_validation=geo_model,
            clinical_support=clinical_support,
            raw_confidence=pred.raw_confidence,
            calibrated_probability=pred.calibrated_probability,
            confidence_status=conf_status,
            evidence_status=status,
            evidence_score=round(score, 4),
            evidence_score_type="weighted_matrix",
            score_is_probability=False,
            contradiction=result.contradiction,
            contradiction_severity=result.severity,
            contradiction_reason=result.reason,
            require_review=True,
            evidence_ids=[model_eid] + ([geo_eid] if geo_model else []),
        )
        f.final_language = self._ddd_language(f, pred, geo)
        return f

    # ------------------------------------------------------------------
    # Spondylolisthesis finding
    # ------------------------------------------------------------------

    def _build_spondy_finding(
        self,
        pred: RawSpondyPrediction,
        geo: Optional[RawGeometry],
        raw: RawCaseInput,
    ) -> Optional[Finding]:
        if pred.slip_percent is None:
            return None

        finding_id = f"F_{pred.level}_SPONDY"
        conf_status = classify_confidence(pred.raw_confidence, self.config)

        geo_model = None
        if geo is not None:
            geo_values: dict[str, Any] = {}
            if geo.offset_ratio is not None:
                geo_values["offset_ratio"] = geo.offset_ratio
            if geo.listhesis_possible is not None:
                geo_values["listhesis_possible"] = geo.listhesis_possible
            geo_eid = self._register(
                EvidenceSource.GEOMETRY, level=pred.level,
                values=geo_values,
                note="Independent vertebral-offset geometric measurement.",
                measurement=True,
            )
            geo_model = GeometricValidation(
                level=pred.level,
                measurements=geo_values,
                supports=bool(geo.listhesis_possible)
                if geo.listhesis_possible is not None else None,
                contradicts=(geo.listhesis_possible is False),
                measurement_quality=geo.measurement_quality,
                landmark_confidence=geo.landmark_confidence,
            )

        model_eid = self._register(
            EvidenceSource.SPONDYLOLISTHESIS_MODEL, level=pred.level,
            values={
                "slip_percent": pred.slip_percent,
                "grade_label": pred.grade_label,
                "raw_confidence": pred.raw_confidence,
                "calibrated_probability": pred.calibrated_probability,
            },
            note="Automated spondylolisthesis slip/grade prediction.",
        )

        claim = pred.grade_label or f"slip of {pred.slip_percent:.1f}%"
        model = ModelPrediction(
            finding_type="SPONDYLOLISTHESIS",
            level=pred.level,
            prediction_value=pred.slip_percent,
            prediction_label=claim,
            grading_system="Meyerding grade (from model)",
            raw_confidence=pred.raw_confidence,
            calibrated_probability=pred.calibrated_probability,
            confidence_status=conf_status,
            model_version=pred.model_version,
        )

        result = self.contradictions_engine.evaluate(model, geo_model, None)

        matrix = self.fuser.build_matrix(model, geo_model, None, result)
        matrix = self.fuser.fuzz(matrix)
        status = self.fuser.resolve_status(matrix, result, model, geo_model)
        score = matrix.score

        f = Finding(
            finding_id=finding_id,
            finding_type=FindingType.SPONDYLOLISTHESIS,
            level=pred.level,
            model_prediction=model,
            geometric_validation=geo_model,
            clinical_support=None,
            raw_confidence=pred.raw_confidence,
            calibrated_probability=pred.calibrated_probability,
            confidence_status=conf_status,
            evidence_status=status,
            evidence_score=round(score, 4),
            evidence_score_type="weighted_matrix",
            score_is_probability=False,
            contradiction=result.contradiction,
            contradiction_severity=result.severity,
            contradiction_reason=result.reason,
            require_review=result.review_required,
            evidence_ids=[model_eid] + ([geo_eid] if geo_model else []),
        )
        f.final_language = self._spondy_language(f, pred, geo)
        return f

    # ------------------------------------------------------------------
    # Disc-space narrowing finding (geometry-derived, Level 2)
    # ------------------------------------------------------------------

    def _build_narrowing_finding(self, g: RawGeometry) -> Finding:
        finding_id = f"F_{g.level}_NARROW"
        geo_eid = self._register(
            EvidenceSource.GEOMETRY, level=g.level,
            values={"relative_space": g.relative_space,
                    "space_narrowed": True},
            note="Independent relative disc-space measurement.",
            measurement=True,
        )
        model = ModelPrediction(
            finding_type="DISC_SPACE_NARROWING",
            level=g.level,
            prediction_label="disc-space narrowing",
            grading_system="relative disc-space ratio",
        )
        geo_model = GeometricValidation(
            level=g.level,
            measurements={"relative_space": g.relative_space,
                          "space_narrowed": True},
            supports=True,
            measurement_quality=g.measurement_quality,
            landmark_confidence=g.landmark_confidence,
        )
        f = Finding(
            finding_id=finding_id,
            finding_type=FindingType.DISC_SPACE_NARROWING,
            level=g.level,
            model_prediction=model,       # no raw confidence: geometric finding
            geometric_validation=geo_model,
            raw_confidence=None,
            calibrated_probability=None,
            confidence_status=ConfidenceStatus.MODERATE_CONFIDENCE,
            evidence_status=EvidenceStatus.SUPPORTED,
            evidence_score=0.0,
            evidence_score_type="geometric_measurement",
            score_is_probability=False,
            contradiction=False,
            contradiction_severity=ContradictionSeverity.NO_CONTRADICTION,
            require_review=False,
            evidence_ids=[geo_eid],
        )
        f.final_language = self._narrowing_language(g)
        return f

    # ------------------------------------------------------------------
    # Clinical context (kept separate from imaging, spec §12)
    # ------------------------------------------------------------------

    def _build_clinical_context(self, raw: RawCaseInput) -> ClinicalContext:
        ctx = raw.clinical_context or {}
        symptoms = raw.symptoms or {}
        clinical_eid = self._register(
            EvidenceSource.CLINICAL,
            values={
                "age": ctx.get("age"),
                "pain_score": ctx.get("pain_score"),
                "pain_duration_years": ctx.get("pain_duration_years"),
            },
            note="Structured clinical history (no free-text PHI stored).",
        )
        correlations = self._symptom_correlations(ctx, symptoms)
        return ClinicalContext(
            age=ctx.get("age"),
            sex=ctx.get("sex", ""),
            pain_score=ctx.get("pain_score"),
            pain_duration_years=ctx.get("pain_duration_years"),
            symptoms=symptoms,
            symptom_correlations=correlations,
            notes=["Clinical findings are separated from objective imaging findings."],
        )

    def _symptom_correlations(self, ctx: dict, symptoms: dict) -> list[str]:
        """Conservative correlation statements (spec §12 / §13 / §28 Test 6)."""
        statements: list[str] = []

        numbness = any(symptoms.get(k, False) for k in
                       ("right_leg_numbness", "left_leg_numbness", "numbness"))
        weakness = symptoms.get("weakness", False)
        radiation = symptoms.get("radiation", "")

        if numbness or weakness or radiation:
            statements.append(
                "The reported lower-limb symptoms (numbness/weakness/"
                "radiation) warrant clinical neurological correlation. The "
                "available automated imaging findings do not independently "
                "establish a specific compressed nerve root."
            )

        if ctx.get("age") is not None or ctx.get("pain_duration_years") is not None:
            statements.append(
                "Chronic back pain history provides relevant clinical context "
                "but does not by itself establish the severity or location of "
                "structural pathology."
            )
        return statements

    # ------------------------------------------------------------------
    # Conservative per-finding language
    # ------------------------------------------------------------------

    def _ddd_language(self, f: Finding, pred: RawDDDPrediction,
                      geo: Optional[RawGeometry]) -> str:
        grade = pred.pfirrmann_grade
        gc = pred.raw_confidence
        if pred.raw_confidence is None:
            gc_part = ""
        else:
            gc_part = f", raw model confidence {pred.raw_confidence:.2f}"

        status = f.evidence_status
        level = f.level
        change_desc = _degeneration_phrase(grade)

        if status == EvidenceStatus.LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT:
            rs = geo.relative_space if geo else None
            rs_part = f", relative disc-space ratio {rs:.3f}" if rs is not None else ""
            return (
                f"Low-confidence automated DDD prediction at {level} "
                f"(Pfirrmann grade ~{grade:.1f}{gc_part}) with independent "
                f"geometric support for focal disc-space narrowing{rs_part}. "
                f"{change_desc.capitalize()} suggested; extent and severity "
                "require radiologist confirmation."
            )
        if status == EvidenceStatus.LOW_CONFIDENCE_UNSUPPORTED:
            return (
                f"Low-confidence automated DDD prediction at {level} "
                f"(Pfirrmann grade ~{grade:.1f}{gc_part}). No independent "
                f"geometric support. {change_desc.capitalize()} cannot be "
                "confirmed without specialist review."
            )
        if status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PROBABLE):
            return (
                f"Automated DDD finding at {level} (Pfirrmann grade ~{grade:.1f}"
                f"{gc_part}) supported by independent geometric disc-space "
                "measurement. Requires radiologist confirmation for severity."
            )
        return (
            f"Automated DDD finding at {level} (Pfirrmann grade ~{grade:.1f}"
            f"{gc_part}). Not independently corroborated; indeterminate "
            "pending radiologist review."
        )

    def _spondy_language(self, f: Finding, pred: RawSpondyPrediction,
                         geo: Optional[RawGeometry]) -> str:
        claim = pred.grade_label or f"slip of {pred.slip_percent:.1f}%"
        level = f.level
        conf_part = f", raw confidence {pred.raw_confidence:.2f}" \
            if pred.raw_confidence is not None else ""
        status = f.evidence_status

        if status == EvidenceStatus.CONTRADICTED:
            return (
                f"AI prediction of {claim} at {level} is not corroborated by "
                f"geometric analysis{conf_part}. Findings are inconsistent "
                "with independent geometric validation and are unreliable "
                "pending specialist review. They must not be treated as a "
                "confirmed grade."
            )
        if status == EvidenceStatus.INDETERMINATE:
            return (
                f"Geometric analysis flags possible vertebral translation at "
                f"{level}, but the available evidence does not independently "
                f"establish {claim}. This finding is indeterminate and should "
                "not be assigned a formal grade without specialist confirmation."
            )
        if status == EvidenceStatus.NOT_CORROBORATED:
            return (
                f"AI prediction of {claim} at {level} could not be corroborated"
                f" because independent geometric validation is unavailable"
                f"{conf_part}. Indeterminate pending radiologist review."
            )
        if status in (EvidenceStatus.SUPPORTED, EvidenceStatus.PROBABLE):
            return (
                f"Automated prediction of moderate translation at {level} "
                "is consistent with geometric analysis. Requires specialist "
                "confirmation before any grade assignment."
            )
        return (
            f"Automated prediction of {claim} at {level} requires specialist "
            "review; status indeterminate."
        )

    def _narrowing_language(self, g: RawGeometry) -> str:
        return (
            f"Focal {g.level} disc-space narrowing is supported by an "
            f"independent geometric measurement (relative disc-space ratio "
            f"{g.relative_space:.3f})."
        )

    # ------------------------------------------------------------------
    # Contradiction packing + overall status
    # ------------------------------------------------------------------

    def _pack_contradiction(self, f: Finding) -> Contradiction:
        cid = f"CON_{f.level}_{f.finding_type.value}"
        model_claim = (
            f.model_prediction.prediction_label
            if f.model_prediction and f.model_prediction.prediction_label
            else str(f.model_prediction.prediction_value)
            if f.model_prediction
            else "n/a"
        )
        return build_contradiction(
            contradiction_id=cid,
            finding_id=f.finding_id,
            ftype=f.finding_type,
            level=f.level,
            result=ContradictionResult(
                contradiction=True,
                severity=f.contradiction_severity,
                reason=f.contradiction_reason,
                rule_id="packed",
                status_hint=f.evidence_status,
                review_required=f.require_review,
            ),
            model_claim=model_claim,
        )

    def _overall_status(self, case: StructuredCase) -> OverallStatus:
        if case.image_quality.status == ImageQualityStatus.INADEQUATE:
            return OverallStatus.INADEQUATE_INPUT

        # A finding survives validation when it is supported, confirmed or
        # an independent (Level-2) geometric measurement.
        surviving = [
            f for f in case.findings
            if f.finding_type == FindingType.DISC_SPACE_NARROWING
            or f.evidence_status in (
                EvidenceStatus.SUPPORTED,
                EvidenceStatus.CONFIRMED,
            )
        ]

        if case.contradictions and not surviving:
            return OverallStatus.HIGH_UNCERTAINTY
        if case.image_quality.status == ImageQualityStatus.LIMITED:
            return OverallStatus.NEEDS_REVIEW
        if case.contradictions:
            return OverallStatus.NEEDS_REVIEW
        if case.low_confidence_findings:
            return OverallStatus.NEEDS_REVIEW
        return OverallStatus.ACCEPTABLE

    def _requires_review(self, case: StructuredCase) -> bool:
        if case.image_quality.status != ImageQualityStatus.ADEQUATE:
            return True
        if case.contradictions:
            return True
        if case.low_confidence_findings:
            return True
        return any(f.require_review for f in case.findings)


# ======================================================================
# Helpers
# ======================================================================

_KNOWN_LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def _pfirrmann_label(grade: float) -> str:
    g = int(round(float(grade)))
    g = max(1, min(5, g))
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}[g]


def _degeneration_phrase(grade: float) -> str:
    """Grade-coupled conservative language (spec §10)."""
    g = float(grade)
    if g <= 2.5:
        return "mild/early degenerative change"
    if g <= 3.5:
        return "moderate degenerative change"
    return "advanced degenerative change"


def build_structured_case(
    raw: RawCaseInput,
    config: Optional[PipelineConfig] = None,
) -> StructuredCase:
    return StructuredCaseBuilder(config).build(raw)