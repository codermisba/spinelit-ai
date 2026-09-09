"""
Longitudinal risk safety (spec §28 scenario 5) + clinical correlation
(scenario 6) + image-quality gating (scenario 7).
"""

from decision_support import run_decision_support


# ------------------------------------------------------------------
# Spec §28 scenario 5: no validated longitudinal model
# ------------------------------------------------------------------

def test_scenario_5_longitudinal_risk_not_available_no_zero():
    result = run_decision_support("scenario-5", use_llm=False)
    lr = result.case.longitudinal_risk
    assert lr.status.value == "NOT_AVAILABLE"
    assert lr.available is False
    assert lr.score is None  # never a synthetic 0.00
    assert "0.00" not in result.report.longitudinal_outlook
    assert "cannot be reliably" in result.report.longitudinal_outlook.lower()


# ------------------------------------------------------------------
# Spec §28 scenario 6: right-leg numbness must not imply nerve root
# ------------------------------------------------------------------

def test_scenario_6_numbness_requires_neurological_correlation():
    result = run_decision_support(
        "scenario-6",
        use_llm=False,
        symptoms={"right_leg_numbness": True},
    )
    text = result.report.text.lower()
    correlation = "\n".join(
        result.case.clinical_context.symptom_correlations).lower()
    assert "neurological correlation" in correlation
    assert "do not independently establish a specific compressed nerve root" \
        in correlation
    assert "nerve root compression" not in text
    assert result.report.clinical_correlation  # rendered in the report


# ------------------------------------------------------------------
# Spec §28 scenario 7: inadequate image quality
# ------------------------------------------------------------------

def test_scenario_7_inadequate_image_quality():
    result = run_decision_support(
        "scenario-7",
        use_llm=False,
        image_quality={
            "status": "inadequate",
            "confidence": 0.90,
            "issues": ["major motion artifact"],
        },
        ddd_predictions=[{
            "level": "L4/L5",
            "pfirrmann_grade": 4.0,
            "raw_confidence": 0.80,   # high conf but image unusable
        }],
    )
    assert result.case.overall_status.value == "INADEQUATE_INPUT"
    assert result.report.overall_status == "INADEQUATE_INPUT"
    # The conservative fallback must still pass validation so the case
    # is returned with a safe report.
    assert result.report_validated
    # No confident diagnosis may be asserted.
    assert "confirmed" not in result.report.disclaimer.lower()


# ------------------------------------------------------------------
# Overall status propagation
# ------------------------------------------------------------------

def test_contradiction_case_overall_status_needs_review():
    result = run_decision_support(
        "status-contradicted",
        use_llm=False,
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
    assert result.case.contradictions
    assert result.case.overall_status.value in ("NEEDS_REVIEW", "HIGH_UNCERTAINTY")
    assert result.case.requires_radiologist_review