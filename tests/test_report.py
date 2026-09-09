"""
Report validator + LLM regeneration/fallback (spec §23-§26).

- The validator rejects forbidden phrasing (e.g. "confirmed grade III").
- `ReportGenerator.generate(attempt_llm=True)` regenerates on invalid
  LLM output, and finally falls back to the deterministic safe report.
- The safe fallback always passes validation and is traceable.
"""

from decision_support import run_decision_support
from pipeline_report import ReportGenerator
from report_validator import ReportValidator
from tests.conftest import bad_report_payload, good_report_payload


def _build_case():
    result = run_decision_support("report-case", use_llm=False,
        ddd_predictions=[{
            "level": "L4/L5",
            "pfirrmann_grade": 4.0,
            "raw_confidence": 0.44,
        }],
        spondy_predictions=[{
            "level": "L2/L3",
            "slip_percent": 30.0,
            "grade_label": "Grade II",
            "raw_confidence": 0.70,
        }],
        geometry=[{
            "level": "L2/L3",
            "offset_ratio": 0.04,
            "listhesis_possible": False,
            "landmark_confidence": 0.80,
        }],
    )
    return result.case


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------

def test_validator_rejects_confirmed_grade_iii(cfg):
    case = _build_case()
    validator = ReportValidator(cfg)
    from evidence_schemas import DecisionSupportReport
    report = DecisionSupportReport(**bad_report_payload())
    check = validator.validate(case, report)
    assert not check.valid
    assert any("unsupported Grade III confirmation" in e for e in check.errors)


def test_validator_accepts_conservative_payload(cfg):
    case = _build_case()
    validator = ReportValidator(cfg)
    from evidence_schemas import DecisionSupportReport
    report = DecisionSupportReport(**good_report_payload())
    check = validator.validate(case, report)
    assert check.valid, check.errors


def test_validator_requires_radiologist_disclaimer(cfg):
    case = _build_case()
    validator = ReportValidator(cfg)
    from evidence_schemas import DecisionSupportReport, TraceStatement
    payload = good_report_payload()
    payload["disclaimer"] = "Auto-generated summary."
    for key in ("objective_imaging_findings", "model_confidence",
                "contradictions", "impression", "recommendations"):
        payload[key] = [
            s for s in payload[key] if "radiologist" not in s.lower()
        ]
    payload["objective_imaging_findings"] = [
        "No automated structural finding met the support criteria."
    ]
    report = DecisionSupportReport(**payload)
    report.traceability = [TraceStatement(statement="s", evidence_ids=["E001"])]
    check = validator.validate(case, report)
    assert not check.valid
    assert any("disclaim" in e.lower() for e in check.errors)


# ------------------------------------------------------------------
# LLM regeneration + fallback
# ------------------------------------------------------------------

def test_invalid_llm_report_triggers_regeneration_then_fallback(cfg, fake_llm_client):
    fake = fake_llm_client([bad_report_payload()])
    case = _build_case()
    result = ReportGenerator(cfg).generate(case, attempt_llm=True)
    assert fake.calls >= 1
    assert not result.llm_used          # fell back to the safe report
    assert result.report_validated      # fallback passes validation
    assert result.report_attempts >= 1
    assert "confirmed grade III" not in result.report.text.lower()
    assert result.report.traceability   # fully traceable


def test_valid_llm_report_is_accepted(cfg, fake_llm_client):
    fake = fake_llm_client([good_report_payload()])
    case = _build_case()
    result = ReportGenerator(cfg).generate(case, attempt_llm=True)
    assert fake.calls == 1
    assert result.llm_used
    assert result.report_validated


def test_no_llm_path_fallback_is_valid_and_traceable(cfg):
    case = _build_case()
    result = ReportGenerator(cfg).generate(case, attempt_llm=False)
    assert not result.llm_used
    assert result.report_validated, result.validation_errors
    assert result.report.traceability
    assert all(
        len(t.statement) and t.evidence_ids for t in result.report.traceability
    )


def test_traceability_maps_to_registered_evidence_ids(cfg):
    case = _build_case()
    registered = set(case.evidence_registry or {})
    assert registered, "case must register evidence"
    for f in case.findings:
        assert all(eid in registered for eid in f.evidence_ids)