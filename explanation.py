"""
explanation.py
==============

Plain-language explanation of what the vision model found.

Two modes, both grounded in the evidence card (never invented):

``deterministic``
    Composed locally from the numeric evidence (levels, Pfirrmann grades,
    severity, calibrated P(DDD), segmentation, Grad-CAM).  Works with no API
    key and is what the hosted demo uses by default.

``gemini`` (or another configured LLM provider)
    The same facts are sent to the LLM with a strict "use only these facts,
    no diagnosis / no treatment advice" instruction.  Any failure degrades to
    the deterministic text and records the error.

The returned dict is what ``/api/analyze`` exposes as ``explanation``::

    {"text": str, "source": "deterministic" | "gemini", "llm_used": bool,
     "model": str | None, "error": str}
"""

from __future__ import annotations

import json
from typing import Any, Optional

PROMPT = """You explain automated lumbar MRI findings to a clinician.

Rules:
- Use ONLY the JSON facts below. Never invent levels, grades, symptoms,
  diagnoses, treatments or follow-up plans.
- 4 to 6 short sentences, plain prose, no markdown, no bullet points.
- Sentence 1: what the vision model detected (levels, detections, landmarks).
- Then state the severity per level: Pfirrmann grade/label, severity word and
  calibrated P(DDD). Lead with the most severe level; group milder levels.
- Mention segmentation and Grad-CAM only if the facts say they are available.
- End by stating these are model estimates from a research prototype that
  require radiologist review, not a diagnosis.

FACTS:
{facts}
"""


def _level_summary(level: dict) -> str:
    label = level.get("pfirrmann_label") or "?"
    grade = level.get("pfirrmann_grade")
    severity = (level.get("severity") or "unknown").lower()
    prob = level.get("calibrated_probability")
    grade_txt = (
        f"grade {grade:.0f}" if isinstance(grade, (int, float))
        else "grade unknown"
    )
    prob_txt = f", P(DDD) {prob:.2f}" if isinstance(prob, (int, float)) else ""
    return f"{level.get('level', '?')} {label} ({grade_txt}, {severity}{prob_txt})"


def build_facts(
    evidence: dict,
    segmentation: Optional[dict] = None,
    gradcam: Optional[dict] = None,
    model_status: Optional[dict] = None,
) -> dict[str, Any]:
    """Reduce the API evidence into the small fact sheet used for text."""
    evidence = evidence or {}
    segmentation = segmentation or {}
    gradcam = gradcam or {}
    model_status = model_status or {}

    levels = [
        {
            "level": f.get("level"),
            "pfirrmann_grade": f.get("pfirrmann_grade"),
            "pfirrmann_label": f.get("pfirrmann_label"),
            "severity": f.get("severity"),
            "calibrated_probability": f.get("calibrated_probability"),
            "localization_quality": f.get("localization_quality"),
        }
        for f in (evidence.get("ddd") or [])
    ]
    levels.sort(
        key=lambda x: (
            x["pfirrmann_grade"] if isinstance(x["pfirrmann_grade"], (int, float)) else -1),
        reverse=True,
    )

    conf = [float(c) for c in (evidence.get("landmark_conf") or [])
            if isinstance(c, (int, float))]
    return {
        "model_version": model_status.get("model_version"),
        "image_size": evidence.get("image_size"),
        "ddd_available": bool(evidence.get("ddd_available", True)),
        "detections": len(evidence.get("detections") or []),
        "detection_source": evidence.get("detection_source") or "none",
        "landmarks": len(evidence.get("landmark_points") or []),
        "mean_landmark_confidence": round(sum(conf) / len(conf), 3) if conf else None,
        "levels": levels,
        "segmentation": {
            "available": bool(segmentation.get("available")),
            "source": segmentation.get("source"),
            "components": len(segmentation.get("boxes") or []),
            "val_dice": (model_status.get("segmentation") or {}).get("val_dice")
            if model_status else segmentation.get("val_dice"),
        },
        "gradcam": {
            "kind": gradcam.get("kind"),
            "level": gradcam.get("level"),
            "grade": gradcam.get("grade"),
            "keypoint": gradcam.get("keypoint"),
        } if gradcam else {},
    }


def deterministic_text(facts: dict[str, Any]) -> str:
    """Compose an honest narrative from the fact sheet alone."""
    parts: list[str] = []

    det = facts.get("detections") or 0
    source = facts.get("detection_source") or "none"
    lm = facts.get("landmarks") or 0
    mc = facts.get("mean_landmark_confidence")
    if det or lm:
        conf_txt = f" with mean landmark confidence {mc:.2f}" if mc else ""
        parts.append(
            f"The vision model detected {det} vertebral/disc blocks "
            f"({source}-derived) and {lm} landmarks{conf_txt} on this "
            f"midsagittal T2 slice."
        )

    levels = facts.get("levels") or []
    if not facts.get("ddd_available", True) or not levels:
        parts.append(
            "Pfirrmann grading is unavailable for this checkpoint, so no "
            "per-level degeneration grade is reported."
        )
    else:
        severe = [l for l in levels
                  if isinstance(l.get("pfirrmann_grade"), (int, float))
                  and l["pfirrmann_grade"] >= 4]
        moderate = [l for l in levels
                    if isinstance(l.get("pfirrmann_grade"), (int, float))
                    and l["pfirrmann_grade"] == 3]
        mild = [l for l in levels
                if isinstance(l.get("pfirrmann_grade"), (int, float))
                and l["pfirrmann_grade"] <= 2]
        described = [l for l in levels if isinstance(l.get("pfirrmann_grade"), (int, float))]

        if described:
            parts.append(
                f"The model graded {len(described)} of {len(levels)} levels: "
                f"{', '.join(_level_summary(l) for l in described)}."
            )
        if severe:
            parts.append(
                f"The most severe change is at {severe[0].get('level')} "
                f"({_level_summary(severe[0])}), with "
                f"{len(severe)} level(s) graded Pfirrmann IV or higher."
                if len(severe) > 1 else
                f"The most severe change is at {severe[0].get('level')} "
                f"({_level_summary(severe[0])})."
            )
        elif moderate:
            parts.append(
                f"Changes are moderate at worst ({_level_summary(moderate[0])}); "
                f"no level reached Pfirrmann IV."
            )
        else:
            parts.append(
                "No level reached Pfirrmann III; findings are limited to mild "
                "degenerative change."
            )

        if mild:
            parts.append(
                "Milder levels: "
                + ", ".join(_level_summary(l) for l in mild) + "."
            )

    seg = facts.get("segmentation") or {}
    if seg.get("available"):
        dice = seg.get("val_dice")
        dice_txt = f" (held-out foreground Dice {dice:.2f})" if isinstance(dice, (int, float)) else ""
        parts.append(
            f"The U-Net segmentation found {seg.get('components', 0)} mask "
            f"components for vertebra, disc and canal{dice_txt}."
        )
    else:
        parts.append("The segmentation model was not available for this request.")

    cam = facts.get("gradcam") or {}
    if cam.get("kind") == "ddd" and cam.get("level"):
        parts.append(
            f"Grad-CAM explains the {cam.get('level')} prediction "
            f"(grade {cam.get('grade')}) and highlights the image regions "
            f"that drove it."
        )
    elif cam.get("kind"):
        parts.append(
            "Grad-CAM highlights the landmark attention used for localization, "
            "not a degeneration grade."
        )

    parts.append(
        "These are model estimates from a research prototype, not a "
        "diagnosis, and require review by a qualified radiologist."
    )
    return " ".join(parts)


def build_explanation(
    evidence: dict,
    segmentation: Optional[dict] = None,
    gradcam: Optional[dict] = None,
    model_status: Optional[dict] = None,
    use_llm: bool = True,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    """Return the grounded explanation payload for ``/api/analyze``."""
    facts = build_facts(evidence, segmentation, gradcam, model_status)
    text = deterministic_text(facts)
    out: dict[str, Any] = {
        "text": text,
        "source": "deterministic",
        "llm_used": False,
        "model": None,
        "error": "",
    }
    if not use_llm:
        return out

    try:
        from llm import make_client
        client = make_client(provider)
        generated = client.complete_text(
            PROMPT.format(facts=json.dumps(facts, indent=2, default=str))
        ).strip()
    except Exception as exc:  # noqa: BLE001 - optional; deterministic stays
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    if not generated:
        out["error"] = "LLM returned an empty explanation."
        return out

    out.update({
        "text": generated,
        "source": (provider or "").strip().lower() or "llm",
        "llm_used": True,
        "model": getattr(client, "model", None),
    })
    # ``source`` should name the backend even when the caller passed nothing.
    if out["source"] in ("", "llm"):
        out["source"] = type(client).__name__.replace("Client", "").lower() or "llm"
    return out
