"""
Evidence layer tests: confidence classification + spec §28 scenarios 1-4.

Scenarios exercised (deterministic path, use_llm=False):
  1. Grade III grade claim, listhesis_possible=True, but no severity
     validation -> INDETERMINATE, never "confirmed grade".
  2. Spondylolisthesis claim with explicitly-measured negative geometry
     (listhesis_possible=False) -> CONTRADICTED (spec §8 R1).
  3. DDD low raw confidence without independent geometry
     -> LOW_CONFIDENCE_UNSUPPORTED.
  4. DDD low raw confidence with independent space-narrowing geometry
     -> LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT.
"""

from confidence import classify_confidence
from decision_support import run_decision_support


def _finding(result, finding_type: str):
    return next(
        f for f in result.case.findings
        if f.finding_type.value == finding_type
    )


# ------------------------------------------------------------------
# Confidence classification
# ------------------------------------------------------------------

def test_classify_confidence_bands(cfg):
    assert classify_confidence(0.80, cfg).value == "HIGH_CONFIDENCE"
    assert classify_confidence(0.69, cfg).value == "MODERATE_CONFIDENCE"
    assert classify_confidence(0.49, cfg).value == "LOW_CONFIDENCE"
    assert classify_confidence(None, cfg).value == "LOW_CONFIDENCE"


# ------------------------------------------------------------------
# Spec §28 scenario 1: Grade III without severity validation
# ------------------------------------------------------------------

def test_scenario_1_grade_iii_not_confirmed_without_severity_validation():
    result = run_decision_support(
        "scenario-1",
        use_llm=False,
        spondy_predictions=[{
            "level": "L1/L2",
            "slip_percent": 45.0,
            "grade_label": "Grade III",
            "raw_confidence": 0.82,
            "calibrated_probability": 0.80,
        }],
        geometry=[{
            "level": "L1/L2",
            "offset_ratio": 0.30,
            "listhesis_possible": True,
            "landmark_confidence": 0.85,
        }],
    )
    f = _finding(result, "SPONDYLOLISTHESIS")
    assert f.evidence_status.value == "INDETERMINATE"
    assert "indeterminate" in f.final_language.lower()
    assert "confirmed grade" not in f.final_language.lower()
    assert f.require_review
    assert result.report_validated


# ------------------------------------------------------------------
# Spec §28 scenario 2: explicit negative geometry falsifies the claim
# ------------------------------------------------------------------

def test_scenario_2_negative_geometry_is_contradicted():
    result = run_decision_support(
        "scenario-2",
        use_llm=False,
        spondy_predictions=[{
            "level": "L4/L5",
            "slip_percent": 25.0,
            "grade_label": "Grade I",
            "raw_confidence": 0.75,
            "calibrated_probability": 0.70,
        }],
        geometry=[{
            "level": "L4/L5",
            "offset_ratio": 0.05,
            "listhesis_possible": False,
            "landmark_confidence": 0.80,
        }],
    )
    f = _finding(result, "SPONDYLOLISTHESIS")
    assert f.evidence_status.value == "CONTRADICTED"
    assert f.contradiction
    assert "not corroborated" in f.final_language.lower()
    assert "must not be treated as a confirmed grade" in f.final_language.lower()
    assert result.case.contradictions


# ------------------------------------------------------------------
# Spec §28 scenario 3: DDD low confidence, no geometry
# ------------------------------------------------------------------

def test_scenario_3_low_conf_ddd_unsupported():
    result = run_decision_support(
        "scenario-3",
        use_llm=False,
        ddd_predictions=[{
            "level": "L3/L4",
            "pfirrmann_grade": 4.0,
            "raw_confidence": 0.44,
            "calibrated_probability": 0.44,
        }],
    )
    f = _finding(result, "DDD")
    assert f.confidence_status.value == "LOW_CONFIDENCE"
    assert f.evidence_status.value == "LOW_CONFIDENCE_UNSUPPORTED"
    assert "no independent geometric support" in f.final_language.lower()


# ------------------------------------------------------------------
# Spec §28 scenario 4: DDD low confidence + independent narrowing
# ------------------------------------------------------------------

def test_scenario_4_low_conf_ddd_with_independent_support():
    result = run_decision_support(
        "scenario-4",
        use_llm=False,
        ddd_predictions=[{
            "level": "L3/L4",
            "pfirrmann_grade": 3.5,
            "raw_confidence": 0.46,
            "calibrated_probability": 0.46,
        }],
        geometry=[{
            "level": "L3/L4",
            "relative_space": 0.70,
            "space_narrowed": True,
            "landmark_confidence": 0.80,
        }],
    )
    f = _finding(result, "DDD")
    assert f.confidence_status.value == "LOW_CONFIDENCE"
    assert f.evidence_status.value == "LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT"
    assert "independent geometric support" in f.final_language.lower()
    assert f.require_review