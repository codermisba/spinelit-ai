# Work-in-progress snapshot — Spine MRI decision-support pipeline

Saved: 2026-09-09. Parts (A)–(K) of the §33 task plan are complete: the
safety-layer modules, the §28 pytest suite (21 tests), the thin-line
anatomy overlay, README updates, and the final success-criteria mapping
are all done and verified.

## What is done (all verified)

New safety layer (additive — existing modules untouched):

| Module | Purpose | Status |
|---|---|---|
| `config.yaml` | all thresholds configurable (`confidence_thresholds`, `geometry`, `evidence_weights`, `reporting`, `validation.reject_patterns/required_content`, `longitudinal`, `audit`) | done |
| `pipeline_config.py` | typed YAML loader + lazy singleton + `with_overrides()` | done |
| `evidence_schemas.py` | pydantic: `ConfidenceStatus`, `EvidenceStatus`, `ContradictionSeverity`, `OverallStatus`, `ModelPrediction`, `GeometricValidation`, `ClinicalSupport`, `Finding`, `Contradiction`, `ImageQuality`, `ClinicalContext`, `LongitudinalRiskResult` (safe null score), `StructuredCase`, `DecisionSupportReport`, `DecisionSupportResult`, `EvidenceEntry` (E001… IDs) | done |
| `confidence.py` | LOW/MODERATE/HIGH classification from config thresholds | done |
| `contradiction_engine.py` | deterministic rules R1–R6 (spec §8, §29) + severity mapping (§9: Grade III vs geometry) | done |
| `evidence_fusion.py` | evidence matrix (§16), weighted score (never a probability), status resolution incl. `LOW_CONFIDENCE_UNSUPPORTED` / `LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT` / `CONTRADICTED` / `NOT_CORROBORATED` / `INDETERMINATE`; explicit-negative geometry → `CONTRADICTED` at any severity (§8 R1) | done |
| `structured_builder.py` | `RawCaseInput` dataclasses, assembles `StructuredCase` (§15), evidence registry, per-finding conservative `final_language`, `overall_status` (ACCEPTABLE/NEEDS_REVIEW/HIGH_UNCERTAINTY/INADEQUATE_INPUT), `requires_radiologist_review`, clinical/symptom separation (§12) | done |
| `longitudinal_risk.py` | NO numeric score when model absent → `score=None`, `status=NOT_AVAILABLE` (§14, §28 Test 5) | done |
| `report_validator.py` | reject-pattern + required-content + structural validation (§23); triggers regeneration; radiologist disclaimer required; empty-objective-findings tolerated for INADEQUATE input | done |
| `audit_log.py` | JSONL audit trail: run id, pipeline/model versions, timestamps, evidence summary; PHI stripped (§24/§31) | done |
| `pipeline_report.py` | §19 system prompt; LLM gets ONLY `StructuredCase` JSON; Gemini `responseSchema` structured output; regenerate up to `llm_max_retries`; deterministic conservative fallback report; **deterministic traceability injected on every accepted report**; self-describing objective-findings section when nothing reached the support threshold | done |
| `decision_support.py` | end-to-end orchestration (§22 modules 9–16), dict input coercion, `run_example()` + example-case builders | done |
| `cli_decision_support.py` | CLI (`--example --pretty --no-llm --provider --out`) | done |
| `llm.py` | `complete_json(prompt, response_schema=...)` now forwards schema to Gemini | done |
| `.gitignore` | added `logs/*.jsonl` | done |

Visualization (new request):

| Item | Purpose | Status |
|---|---|---|
| `utils.draw_thin_anatomy_overlay` | 1 px thin-line schematic: vertebra-body outlines + column axis + horizontal offset lines, derived **only** from the 10 detected keypoints | done |
| `utils.draw_landmarks(..., anatomy_lines=True)` | renders the thin-line overlay under the existing landmark markers (default on) | done |
| Legend | adds "Vertebra outline (thin lines)" entry | done |

## Test suite (part K)

- Files: `tests/conftest.py` (fixtures, FakeLLMClient, bad/good LLM payloads),
  `tests/test_evidence.py`, `tests/test_longitudinal_clinical.py`,
  `tests/test_report.py`, `tests/test_drawing.py`.
- `& .venv\Scripts\python.exe -m pytest tests/ -q` → **21 passed**.
- §28 scenarios covered: 1 (Grade III + geometry possible → INDETERMINATE, not
  "confirmed grade"), 2 (negative geometry → CONTRADICTED), 3 (DDD low conf,
  no geometry → LOW_CONFIDENCE_UNSUPPORTED), 4 (DDD low conf + narrowed →
  LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT), 5 (no longitudinal model → score
  None, no 0.00), 6 (numbness → neurological correlation, no nerve-root
  claim), 7 (inadequate image quality → INADEQUATE_INPUT, safe report).
- Plus: confidence bands, task 6 trailing; validator reject patterns,
  radiologist-disclaimer rule, regeneration→fallback on invalid LLM output,
  valid-LLM acceptance with computed traceability, fallback passes validator +
  is fully traceable, evidence IDs resolve against the registry, thin-line
  drawing (width 1, landmark-derived, degenerate-keypoint safety).

## Verified behaviour (smoke, use_llm=False)

- Worked example → `overall_status=NEEDS_REVIEW`, `report_validated=True`,
  traceability non-empty, ~0.5 s after warm import.
- DDD: L1/L2, L2/L3, L4/L5, L5/S1 → `LOW_CONFIDENCE`; L4/L5 also
  `LOW_CONFIDENCE_WITH_INDEPENDENT_SUPPORT` (relative_space 0.734); L3/L4
  `MODERATE` → provisional.
- Spondylolisthesis: L1/L2 → INDETERMINATE (moderate contra°, grade III
  downgraded); L2/L3, L3/L4, L4/L5 (geometry false) → `CONTRADICTED` MAJOR;
  L5/S1 (no geometry) → `NOT_CORROBORATED`.
- Longitudinal `score=None`, `status=NOT_AVAILABLE`.
- Recognized-after-fix: a case with only indeterminate/contradicted findings
  now yields a self-describing objective section and `report_validated=True`.

## Real LLM path (Gemini)

- `gemini-3.6-flash` JSON call verified once live. Full report generation hit
  transient HTTP 503 → pipeline retried and fell back safely
  (`report_validated=True`, `llm_used=False`). No hallucination risk. Tests
  never depend on the live API.

## Gotchas for resuming

- Use `& .venv\Scripts\python.exe` (plain `python` not on PATH).
- `.venv` now has pytest installed.
- `GEMINI_API_KEY` set in `.env` (len 39); never print it.
- Cold real-LLM attempt can take ~60 s before fallback; tests use
  `use_llm=False` or the fake client.
- Nothing is committed to git (user hasn't asked). `logs/audit_*.jsonl`
  gitignored.