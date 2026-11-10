# Spine MRI API — Frontend Integration Guide

Live service: **https://spine-mri-api.onrender.com**

Interactive API docs (Swagger): **https://spine-mri-api.onrender.com/docs**
Bundled demo UI: **https://spine-mri-api.onrender.com/ui**
This guide (always current): **https://spine-mri-api.onrender.com/integration**

> Research prototype for lumbar-spine MRI analysis. Outputs are model
> estimates, **not a diagnosis**, and must be reviewed by a qualified
> radiologist. Do not present the output as medical advice.

---

## 1. What the API does

Upload one midsagittal T2 lumbar MRI image. The API returns, in a single
response:

1. **Landmark localization** — 10 keypoints (5 vertebrae + 5 discs) with
   confidences.
2. **Detections** — labelled vertebra/disc blocks with boxes and a
   `detection_source` (`landmark`, `detector`, `segmentation` or `mock`).
3. **Per-level DDD / Pfirrmann grading** — grade 1–5, label (`I`–`V`),
   severity word, and calibrated P(DDD) for each disc level.
4. **U-Net segmentation** — mask components plus a ready-to-display
   overlay PNG (vertebra / disc / spinal canal).
5. **Grad-CAM** — heatmap overlay showing what drove the most severe
   prediction (or the landmark attention when Grad-CAM is localization-based).
6. **Plain-language explanation** — an LLM narrative when a Gemini key is
   configured, otherwise a deterministic narrative built from the exact same
   model evidence.
7. **Decision-support report** — validated text report, impression and
   traceability.

All images come back as **data URLs** (`data:image/png;base64,…`), so you do
not need a second request to render them.

---

## 2. Hosting, CORS and cold starts

| Item | Value |
| --- | --- |
| Base URL | `https://spine-mri-api.onrender.com` |
| CORS | `*` by default, `GET/POST/OPTIONS` allowed. Set the host's `CORS_ORIGINS` env var (comma-separated) to lock this down. |
| Auth | Optional. If the host sets `API_KEYS`, send `X-Api-Key: <key>` or `Authorization: Bearer <key>`. |
| Cold start | The free Render instance sleeps after ~15 min idle. The first request can take **30–90 s** while the 107 MB model downloads. Use a **120 s** client timeout. |
| Warm analyze | ~5–15 s on the free CPU instance (single image). |

Call `GET /api/health` on page load (or before enabling the submit button) and
show a "waking the model…" state until it succeeds.

```js
const API = "https://spine-mri-api.onrender.com";

async function warmUp() {
  const r = await fetch(`${API}/api/health`, { cache: "no-store" });
  const health = await r.json();
  return health.vision?.loaded === true;
}
```

---

## 3. Endpoints

### `GET /api/health`

```json
{
  "status": "ok",
  "version": "2.0.0",
  "vision": {
    "loaded": true,
    "model_version": "20261109-145048",
    "ddd_available": true,
    "image_size": [256, 256],
    "tasks": { "localization": true, "ddd": true, "seg": false },
    "segmentation": { "loaded": true, "val_dice": 0.486 }
  },
  "registry": { "...": "internal weight-resolution details" },
  "mock_fallback_allowed": false,
  "gemini_configured": false
}
```

Use `vision.loaded` to gate analysis, `vision.ddd_available` to hide the
per-level table when the checkpoint has no DDD head, and `gemini_configured`
to explain which kind of explanation you will receive.

### `POST /api/analyze`

`multipart/form-data`:

| Field | Required | Type | Meaning |
| --- | --- | --- | --- |
| `file` | **yes** | file | MRI image (JPG/PNG, converted to RGB). |
| `explain` | no | `"true"` / `"false"` | Compute Grad-CAM. Default `true`. |
| `use_llm` | no | `"true"` / `"false"` | Ask for the LLM explanation. Default `true`; falls back to deterministic text when no key/LLM is available. |
| `provider` | no | string | LLM backend (`gemini`, `ollama`). Empty = server config. |
| `case_id` | no | string | Your case/patient reference. Echoed in `case.case_id`. |
| `clinical` | no | JSON string | Optional context: `{"age": 54, "sex": "F", "symptoms": {...}, "image_quality": "good"}`. Send `{}` if unknown. |
| `spondy_predictions` | no | JSON string | Optional list of spondylolisthesis predictions. Send `[]` if unused. |
| `mock_vision` | no | `"true"` / `"false"` | **Always send `"false"` in production.** Only `true` returns a clearly-labelled synthetic demo. |

```js
async function analyze(file, { useLlm = true, clinical = {} } = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("explain", "true");
  form.append("use_llm", useLlm ? "true" : "false");
  form.append("clinical", JSON.stringify(clinical));
  form.append("spondy_predictions", "[]");
  form.append("mock_vision", "false"); // never show synthetic output as real

  const r = await fetch(`${API}/api/analyze`, {
    method: "POST",
    body: form,
    // headers: { "X-Api-Key": "<key>" }, // only if the host sets API_KEYS
  });
  const body = await r.json();
  if (!r.ok) throw new Error(`${r.status}: ${body.detail || "request failed"}`);
  return body;
}
```

#### Response shape (`AnalyzeResponse`)

Top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `case` | object | Structured case: `case_id`, `overall_status`, findings, etc. |
| `report` | object \| null | Validated report: `impression`, `confidence_limitations`, `recommendations`, `disclaimer`, `traceability`. |
| `report_validated` | bool | Whether the report passed the safety validator. |
| `validation_errors` | string[] | Validator messages, if any. |
| `llm_used` | bool | Whether the **report** used an LLM. |
| `errors` | string[] | Non-fatal pipeline/LLM errors. |
| `evidence` | object | Machine evidence (see below). |
| `input_image_data_url` | string | The uploaded MRI, RGB PNG data URL. |
| `annotated_image_data_url` | string | MRI with blocks + landmarks drawn. |
| `segmentation_image_data_url` | string | U-Net overlay PNG data URL (empty if unavailable). |
| `gradcam_image_data_url` | string | Grad-CAM overlay PNG data URL (empty if disabled/unavailable). |
| `segmentation` | object | `{available, source, version, boxes?}`. |
| `gradcam` | object | `{kind: "ddd"\|"localization", level?, grade?, keypoint?, confidence?, note?, error?}`. `kind: "ddd"` means Grad-CAM explains the most severe predicted level; `localization` is attention only, **not** a grade explanation. |
| `explanation` | object | `{text, source, llm_used, model, error}` — see below. |
| `detections` | array | `{label, class_name, source, confidence, box}`. |
| `detection_source` | string | `landmark`, `detector`, `segmentation`, `mock` or `none`. |
| `ddd_available` | bool | Per-level grading available for this checkpoint. |
| `model_status` | object | Same structure as `health.vision`. |
| `mock_vision` | bool | **Must be `false` for real output.** Never display a success state if this is `true`. |

`evidence`:

```json
{
  "image_processed": true,
  "image_name": "100_t2.jpg",
  "image_size": [256, 256],
  "landmark_points": [[x, y], "..."],
  "landmark_conf": [0.98, "..."],
  "detections": ["..."],
  "detection_source": "landmark",
  "ddd_available": true,
  "ddd": [
    {
      "level": "L2/3",
      "pfirrmann_grade": 4.0,
      "pfirrmann_label": "IV",
      "severity": "Severe",
      "class_probabilities": [0.02, 0.05, 0.18, 0.61, 0.14],
      "raw_confidence": 0.61,
      "calibrated_probability": 0.927,
      "localization_quality": 0.96,
      "evidence": "human-readable numeric evidence"
    }
  ],
  "geometric_indicators": [
    { "level": "L1/2", "relative_space": 0.93, "narrowed": false,
      "offset_ratio": 0.02, "offset_flag": false }
  ],
  "segmentation": { "available": true, "source": "unet", "boxes": ["..."] },
  "notes": ["..."]
}
```

`explanation`:

```json
{
  "text": "The vision model detected ... These are model estimates from a research prototype, not a diagnosis, and require review by a qualified radiologist.",
  "source": "gemini",
  "llm_used": true,
  "model": "gemini-2.0-flash",
  "error": ""
}
```

- `source` is `"gemini"` (or another provider name) when the LLM produced the
  text, otherwise `"deterministic"`.
- `llm_used === false` means the text is the deterministic, evidence-grounded
  fallback. It is still real model output; only the prose is not LLM-generated.
- If `error` is non-empty, show the deterministic text **and** a small note
  such as "LLM unavailable — showing the model-based summary."

---

## 4. Rendering the result

```js
function render(body) {
  if (body.mock_vision) {
    // Never show this as a real prediction.
    showBanner("Synthetic demo output — not a real model prediction", "warning");
  }

  inputImg.src = body.input_image_data_url;
  segImg.src = body.segmentation_image_data_url;
  annotatedImg.src = body.annotated_image_data_url;
  gradcamImg.src = body.gradcam_image_data_url;

  explanationEl.textContent = body.explanation.text;
  explanationBadge.textContent =
    body.explanation.llm_used ? `LLM · ${body.explanation.model}` : "model-based summary";

  for (const level of body.evidence.ddd) {
    addRow({
      level: level.level,
      pfirrmann: `${level.pfirrmann_label} (${level.pfirrmann_grade})`,
      severity: level.severity,
      pDDD: level.calibrated_probability?.toFixed(2),
      localization: level.localization_quality?.toFixed(2),
    });
  }
}
```

Image guidance:

- Show the **input MRI and the segmentation overlay side by side**, both
  using the same object-fit so the anatomy lines up.
- The overlay is already colour-blended: cyan = vertebra, yellow = disc,
  green = spinal canal.
- The gallery panels (`/api/segmentation/samples`) are
  `MRI | ground truth | prediction` composites and are safe to show as
  held-out evidence of segmentation quality.

### Severity mapping

| Pfirrmann | Label | Typical severity |
| --- | --- | --- |
| 1 | I | Normal |
| 2 | II | Mild |
| 3 | III | Moderate |
| 4 | IV | Severe |
| 5 | V | Severe |

Always render the API's own `severity` string rather than recomputing it.

---

## 5. Endpoints for the demo assets

### `GET /api/segmentation/samples`

```json
{
  "count": 4,
  "dirs": ["...", "..."],
  "samples": [
    {
      "name": "case_001",
      "file": "case_001.png",
      "source": "samples",
      "size_bytes": 97248,
      "url": "/api/segmentation/samples/case_001.png"
    }
  ]
}
```

`url` is **relative** to the API base URL. Open
`${API}${sample.url}` for the PNG.

### `GET /api/segmentation/samples/{filename}`

Returns `image/png` with `Cache-Control: no-store`. Invalid names return
`400`; unknown files return `404`.

### `GET /integration`

This guide as `text/markdown; charset=utf-8`, never cached. Link it from your
app's help/about section.

### `GET /` and `GET /ui`

`/` returns service metadata and a `frontend_integration` pointer. `/ui` is a
working reference frontend you can open to verify the live deployment.

---

## 6. Error handling

| Status | When | What to show |
| --- | --- | --- |
| `400` | Corrupt/unsupported image, or invalid `clinical` / `spondy_predictions` JSON | "That file could not be read. Upload a JPG or PNG MRI." |
| `401` | `API_KEYS` is configured and the key is missing/wrong | "Not authorized." Do not retry. |
| `422` | Missing `file` part (FastAPI validation) | "Please choose an image first." |
| `503` | No vision model loaded on the host | "The model is not available right now." Include `body.model_status` in diagnostics. |
| `5xx` / network | Host error, timeout, cold-start failure | Offer retry. First request may need up to 120 s. |

Show `body.errors` and `body.validation_errors` in a collapsible diagnostics
panel, not in the main result.

---

## 7. Integration checklist

- [ ] Call `/api/health` on load; disable **Analyze** until `vision.loaded`.
- [ ] Use a 120 s timeout for the first request (cold start).
- [ ] Always send `mock_vision=false`; treat `mock_vision=true` as an error in production.
- [ ] Display `input_image_data_url` next to `segmentation_image_data_url`.
- [ ] Show `annotated_image_data_url` and, when present, `gradcam_image_data_url`.
- [ ] Render `explanation.text`; badge it `LLM` vs `model-based` from `explanation.source` / `llm_used`.
- [ ] Render the `evidence.ddd` table; hide it and show the reason when `ddd_available` is false.
- [ ] Show the report `impression` and `disclaimer`; keep the research-prototype disclaimer visible.
- [ ] If the host sets `API_KEYS`, send the key from a server-side proxy — never ship it in public JS.
- [ ] Add a retry button for network/5xx errors.
- [ ] Do not derive treatment recommendations from the output.

---

## 8. Minimal HTML example

```html
<input id="file" type="file" accept="image/*">
<button id="go">Analyze</button>
<pre id="out"></pre>

<script>
const API = "https://spine-mri-api.onrender.com";

document.getElementById("go").onclick = async () => {
  const file = document.getElementById("file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("use_llm", "true");
  form.append("explain", "true");
  form.append("mock_vision", "false");
  form.append("clinical", "{}");

  const r = await fetch(`${API}/api/analyze`, { method: "POST", body: form });
  const body = await r.json();
  document.getElementById("out").textContent = r.ok
    ? body.explanation.text
    : `${r.status}: ${body.detail}`;
};
</script>
```

---

## 9. Notes for the backend/host owner

- The explanation is `deterministic` unless `GEMINI_API_KEY` is set on the
  host. Add it under the Render service → **Environment** to switch
  `explanation.source` to `gemini` with no frontend change.
- Model weights live in the private HF repo `MisbaShaikh/spine-severity`;
  the host needs `HF_TOKEN` and `MODEL_ID` (see `render.yaml`).
- To pin a different checkpoint, set `HF_HUB_REVISION` on the host.
- `CORS_ORIGINS` defaults to `*`. Set it to your deployed frontend origin
  before launch.
- `API_KEYS` is optional. If enabled, proxy requests through your backend so
  the key is never exposed in browser JavaScript.
