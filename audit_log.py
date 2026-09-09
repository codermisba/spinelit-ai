"""
audit_log.py
============

Audit logging / traceability (spec §24 and §31).

Writes one JSON line per pipeline run to logs/audit_<date>.jsonl with:

    - run id, timestamps
    - pipeline version, model versions
    - input identifiers (case id / image path) — NOT raw PHI unless
      explicitly enabled in config (`audit.include_patient_identifier`)
    - list of findings + evidence IDs + statuses + contradictions
    - LLM provider/model, number of attempts
    - report validation result

No medical free-text (symptom narratives) is written to the audit log by
default.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pipeline_config import PipelineConfig


class AuditLogger:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.get()
        self._enabled = self.config.audit.enabled
        self._log_dir = Path(self.config.audit.log_dir)
        self._include_phi = self.config.audit.include_patient_identifier

    @staticmethod
    def new_run_id() -> str:
        return f"run-{uuid.uuid4().hex[:12]}"

    def _file(self) -> Path:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self._log_dir / f"audit_{date}.jsonl"

    def _sanitize(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        if not self._include_phi:
            payload.pop("patient_names", None)
            payload.pop("patient_id", None)
            payload.pop("mrn", None)
            payload.pop("free_text", None)
            payload.pop("symptoms_free_text", None)
            for f in payload.get("findings", []):
                if isinstance(f, dict):
                    f.pop("clinical_note", None)
        return payload

    def log(
        self,
        *,
        run_id: str,
        case_id: str,
        pipeline_version: str,
        input_identifiers: Optional[dict[str, Any]] = None,
        model_versions: Optional[dict[str, str]] = None,
        summary: Optional[dict[str, Any]] = None,
    ) -> str:
        """Write one audit record. Returns run_id."""
        if not self._enabled:
            return run_id

        self._log_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": pipeline_version,
            "case_id": case_id,
            "input_identifiers": input_identifiers or {},
            "model_versions": model_versions or {},
            "summary": summary or {},
        }
        record = self._sanitize(record)

        with open(self._file(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return run_id


def build_audit_summary(case, report) -> dict[str, Any]:
    """Compact, PHI-free summary extracted from structured evidence."""
    findings = []
    for f in case.findings:
        findings.append({
            "finding_id": f.finding_id,
            "finding_type": f.finding_type.value,
            "level": f.level,
            "confidence_status": f.confidence_status.value,
            "evidence_status": f.evidence_status.value,
            "evidence_score": round(f.evidence_score, 3),
            "score_is_probability": f.score_is_probability,
            "contradiction": f.contradiction,
            "contradiction_severity": f.contradiction_severity.value,
            "evidence_ids": f.evidence_ids,
        })
    return {
        "overall_status": case.overall_status.value,
        "requires_radiologist_review": case.requires_radiologist_review,
        "contradictions": [
            {
                "id": c.contradiction_id,
                "level": c.level,
                "type": c.finding_type.value,
                "severity": c.severity.value,
            }
            for c in case.contradictions
        ],
        "findings": findings,
        "longitudinal_risk_available": case.longitudinal_risk.available,
        "report_validated": bool(report is not None),
        "report_has_numeric_risk": (
            report is not None and case.longitudinal_risk.score is not None
        ),
    }