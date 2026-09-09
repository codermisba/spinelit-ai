"""
pipeline_config.py
==================

Loads `config.yaml` (the evidence/validation configuration for the
decision-support safety layer) into typed, validated dataclasses.

The legacy `config.py` is intentionally untouched — existing modules keep
importing from it. New validation modules read their thresholds from here.

All thresholds in `config.yaml` are configurable at runtime; nothing
medical-safety-relevant is hardcoded.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when the YAML configuration is missing or invalid."""


# ----------------------------------------------------------
# Typed settings
# ----------------------------------------------------------

class ConfidenceThresholds:
    def __init__(self, data: dict):
        self.low = float(data.get("low", 0.50))
        self.moderate = float(data.get("moderate", 0.70))
        if not (0.0 <= self.low < self.moderate <= 1.0):
            raise ConfigError(
                f"Invalid confidence thresholds: low={self.low}, "
                f"moderate={self.moderate}. Require 0 <= low < moderate <= 1."
            )


class GeometrySettings:
    def __init__(self, data: dict):
        # threshold currently in use; marked as requiring clinical validation
        self.listhesis_offset_ratio_threshold = float(
            data.get("listhesis_offset_ratio_threshold", 0.25))
        self.relative_space_threshold = float(
            data.get("relative_space_threshold", 0.85))
        self.min_landmark_confidence = float(
            data.get("min_landmark_confidence", 0.50))


class EvidenceWeights:
    def __init__(self, data: dict):
        self.model_support = float(data.get("model_support", 1.0))
        self.geometric_support = float(data.get("geometric_support", 1.5))
        self.clinical_support = float(data.get("clinical_support", 0.5))
        self.model_contradiction = float(data.get("model_contradiction", -1.0))
        self.geometric_contradiction = float(
            data.get("geometric_contradiction", -2.0))
        self.confidence_multiplier = float(
            data.get("confidence_multiplier", 0.75))


class EvidenceScoreThresholds:
    def __init__(self, data: dict):
        self.strongly_supported = float(data.get("strongly_supported", 2.2))
        self.supported = float(data.get("supported", 1.0))
        self.provisional = float(data.get("provisional", 0.0))
        self.indeterminate = float(data.get("indeterminate", -0.5))


class ReportingSettings:
    def __init__(self, data: dict):
        self.allow_causal_language = bool(data.get("allow_causal_language", False))
        self.allow_numeric_risk_without_model = bool(
            data.get("allow_numeric_risk_without_model", False))
        self.require_radiologist_disclaimer = bool(
            data.get("require_radiologist_disclaimer", True))
        self.max_impression_points = int(data.get("max_impression_points", 5))
        self.llm_max_retries = int(data.get("llm_max_retries", 2))
        self.require_low_confidence_disclosure = bool(
            data.get("require_low_confidence_disclosure", True))


class ValidationSettings:
    def __init__(self, data: dict):
        self.require_geometry_for_spondylolisthesis_confirmation = bool(
            data.get("require_geometry_for_spondylolisthesis_confirmation", True))
        self.require_level_validation = bool(data.get("require_level_validation", True))
        self.reject_unsupported_grade = bool(data.get("reject_unsupported_grade", True))
        self.severe_slip_percent = float(data.get("severe_slip_percent", 40.0))
        self.reject_patterns = data.get("reject_patterns", [])
        self.required_content = data.get("required_content", [])


class LongitudinalSettings:
    def __init__(self, data: dict):
        self.default_horizon_years = int(data.get("default_horizon_years", 5))
        self.risk_model_available = bool(data.get("risk_model_available", False))
        self.risk_model_path = data.get("risk_model_path", "")


class AuditSettings:
    def __init__(self, data: dict):
        self.enabled = bool(data.get("enabled", True))
        self.log_dir = data.get("log_dir", "logs")
        self.include_patient_identifier = bool(
            data.get("include_patient_identifier", False))


class PipelineSettings:
    def __init__(self, data: dict):
        self.version = str(data.get("version", "unknown"))
        self.name = str(data.get("name", "spine-decision-support"))


class PipelineConfig:
    """Full validated runtime configuration (loaded once, lazily)."""
    _instance: "PipelineConfig | None" = None

    def __init__(self, data: dict):
        self.raw: dict = data
        self.pipeline = PipelineSettings(data.get("pipeline", {}))
        self.confidence = ConfidenceThresholds(data.get("confidence_thresholds", {}))
        self.geometry = GeometrySettings(data.get("geometry", {}))
        self.evidence_weights = EvidenceWeights(data.get("evidence_weights", {}))
        self.evidence_score = EvidenceScoreThresholds(
            data.get("evidence_score_thresholds", {}))
        self.reporting = ReportingSettings(data.get("reporting", {}))
        self.validation = ValidationSettings(data.get("validation", {}))
        self.longitudinal = LongitudinalSettings(data.get("longitudinal", {}))
        self.audit = AuditSettings(data.get("audit", {}))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping: {path}")
        return cls(data)

    @classmethod
    def get(cls) -> "PipelineConfig":
        """Lazily load the shared configuration singleton."""
        if cls._instance is None:
            cls._instance = cls.load()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    def with_overrides(self, overrides: dict[str, Any]) -> "PipelineConfig":
        """Return a deep-copied config with dotted-key runtime overrides."""
        merged = copy.deepcopy(self.raw)
        for key, value in (overrides or {}).items():
            node = merged
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return type(self)(merged)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
    return PipelineConfig.load(path)