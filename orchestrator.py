"""
orchestrator.py
===============

The Agent Orchestrator: coordinator for the end-to-end agentic pipeline.

Runs the agents in a dependency-ordered sequence over ONE patient case
and returns a single `PipelineResult` bundle for the UI / CLI / export:

    1. Symptom Extraction Agent        (LLM)  free-text -> StructuredSymptoms
    2. Vision Engine (Vision Agent)    (model) image -> EvidenceCard (+ calibration)
    3. Fusion Reasoning Agent          (LLM)  clinical+symptoms+evidence -> interpretation
    4. Verification / Critique Agent   (LLM)  audits interpretation (accuracy loop)
    5. Longitudinal Assessment Agent   (model+LLM) progression outlook
    6. Report Writer Agent             (LLM)  verified evidence -> radiology report

Graceful degradation: each agent is independent. If the LLM is not
configured, the pipeline still returns all numeric vision evidence +
model status so the system is always usable in a "degraded" mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import (
    BEST_MODEL,
    OUTPUT_DIR,
    PROGRESSION_HORIZON_YEARS,
    VERIFY_ITERATIONS,
)
from schemas import (
    ClinicalInterpretation,
    EvidenceCard,
    FinalReport,
    LongitudinalAssessment,
    PatientInput,
    PipelineResult,
    StructuredSymptoms,
    VerificationResult,
)
from vision_engine import VisionEngine
from llm import LLMConfigError

from agents import (
    fusion_agent,
    longitudinal_agent,
    report_writer_agent,
    symptom_extraction_agent,
    verifier_agent,
)
from agents.agent_base import AgentError


class AgenticPipeline:
    """Runs the full agentic pipeline for one case."""

    def __init__(self, vision_engine: Optional[VisionEngine] = None):
        self.vision = vision_engine or VisionEngine(str(BEST_MODEL))
        self._cached_features: Optional[object] = None
        self._cached_image = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, patient: PatientInput, pil_image=None,
            provider: str | None = None) -> PipelineResult:
        result = PipelineResult(patient=patient.to_dict())
        result.model_status = {
            "vision": self.vision.status(),
            "calibration_source": self._calibration_source(),
            "llm_provider": provider,
        }

        # ---- 2. Vision Engine ----
        composed = self._compose_image(patient, pil_image)
        image_name = Path(patient.image_path).name if patient.image_path else ""
        if composed is not None:
            card = self.vision.analyze(composed, image_name=image_name)
            self._cached_image = composed
            try:
                self._cached_features = self.vision.extract_features(composed)
            except Exception:  # noqa: BLE001
                self._cached_features = None
            result.annotated_image_path = str(
                self._save_annotated(composed, patient)
            )
        else:
            card = EvidenceCard(image_processed=False, image_name=image_name,
                                notes=["No image provided / not found."])
        result.evidence = card

        # ---- 1. Symptom Extraction Agent (LLM) ----
        try:
            result.symptoms = symptom_extraction_agent(patient.symptoms_text,
                                                       provider=provider)
        except (AgentError, LLMConfigError) as exc:
            result.errors.append(f"symptom_agent: {exc}")
            result.symptoms = StructuredSymptoms(confidence=0.0)

        # ---- 3. Fusion Reasoning Agent (LLM) ----
        try:
            result.interpretation = fusion_agent(
                patient, result.symptoms, card, provider=provider
            )
        except (AgentError, LLMConfigError) as exc:
            result.errors.append(f"fusion_agent: {exc}")
            result.interpretation = ClinicalInterpretation(
                summaries={"disc_degenerative_disease":
                           "(LLM reasoning unavailable — see evidence tables.)",
                           "spondylolisthesis":
                           "(LLM reasoning unavailable — see evidence tables.)"}
            )

        # ---- 4. Verification / Critique Agent (accuracy loop) ----
        result.verification = self._run_verification(patient, result, provider)

        # ---- 5. Longitudinal Assessment ----
        try:
            result.longitudinal = longitudinal_agent(
                patient, result.symptoms, card, result.interpretation,
                image_features=self._cached_features,
                years_ahead=patient.years_ahead or PROGRESSION_HORIZON_YEARS,
                provider=provider,
            )
        except (AgentError, LLMConfigError) as exc:
            result.errors.append(f"longitudinal_agent: {exc}")

        # ---- 6. Report Writer Agent ----
        try:
            result.report = report_writer_agent(
                patient, result.symptoms, card, result.interpretation,
                result.verification, result.longitudinal, provider=provider,
            )
        except (AgentError, LLMConfigError) as exc:
            result.errors.append(f"report_agent: {exc}")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _calibration_source() -> str:
        from calibration import get_calibrators
        return get_calibrators().source

    @staticmethod
    def _compose_image(patient: PatientInput, pil_image):
        if pil_image is not None:
            return pil_image
        if patient.image_path:
            path = Path(patient.image_path)
            if path.exists():
                from PIL import Image
                return Image.open(path).convert("RGB")
        return None

    def _save_annotated(self, image, patient: PatientInput) -> Path:
        OutputDir = OUTPUT_DIR
        OutputDir.mkdir(exist_ok=True)
        stem = Path(patient.image_path).stem if patient.image_path else "case"
        out = OutputDir / f"{stem}_agentic_annotated.jpg"
        try:
            return self.vision.save_annotated(image, out)
        except Exception:  # noqa: BLE001
            return out

    def _run_verification(self, patient, result: PipelineResult,
                          provider) -> VerificationResult:
        if result.interpretation is None:
            return VerificationResult(overall_status="rejected",
                                      summary="No interpretation to verify.")
        current = result.interpretation
        output = None
        for _ in range(max(1, VERIFY_ITERATIONS)):
            try:
                output = verifier_agent(
                    patient, result.symptoms, result.evidence, current,
                    provider=provider,
                )
            except (AgentError, LLMConfigError) as exc:
                result.errors.append(f"verifier_agent: {exc}")
                output = VerificationResult(
                    overall_status="needs_revision",
                    summary="Verifier unavailable; report marked unverified.",
                )
                break
            # In a single-loop practically we stop at the first audit. The
            # loop exists so VERIFY_ITERATIONS could drive additional passes.
        return output


def run_pipeline(patient: PatientInput, pil_image=None,
                 provider: str | None = None,
                 pipeline: Optional[AgenticPipeline] = None) -> PipelineResult:
    pipeline = pipeline or AgenticPipeline()
    return pipeline.run(patient, pil_image=pil_image, provider=provider)
