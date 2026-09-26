# Deploying the Spine API (Hugging Face Spaces / Render)

This guide turns `api_server.py` into a live, public API that a frontend can
call over HTTPS. The key design goal: **you can retrain / improve the model
and update the API without a code change** — weights are versioned in a model
registry and each checkpoint carries its own training metadata.

## How it stays working after retraining

1. `train.py` saves a `meta` block in every checkpoint
   (`image_size`, `disc_feature_stage`, `version`, `model_name`, ...).
2. `utils.load_model` reads that meta and applies it at load time, so a
   model trained at 512px loads and runs correctly even though the config
   default is 256px — no config edit, no restart required for the image
   pipeline to match the checkpoint.
3. `weights.resolve_checkpoint()` picks the active weights from
   (highest priority first):
   - `VISION_CHECKPOINT` env var — a local path **or** an HF Hub repo id
   - `MODEL_ID` env var — an HF Hub repo id (e.g. `acme/spine-severity`)
   - `checkpoints/best_model.pth` (trained locally)
   - `checkpoints/release_model.pth` (committed fp16 artifact)
4. Retraining loop:

   ```
   train on Colab  ->  checkpoints/best_model.pth
   python publish_model.py --repo acme/spine-severity --with-calibration
   (restart the host; the API downloads the new weights and serves them)
   ```

   For a pinned, reproducible deploy pass `--rev v2` and set
   `HF_MODEL_REVISION=v2` on the host.

## One-file pick

- `api_server.py` — FastAPI app (endpoints `/`, `/api/health`, `/api/analyze`)
- `Dockerfile` — CPU-torch image (works on HF Spaces **and** Render)
- `weights.py` — HF Hub model registry + auto-download
- `publish_model.py` — upload retrained weights to the Hub

---

## Option A — Hugging Face Spaces (recommended)

1. Create a Space at <https://huggingface.co/spaces>:
   - SDK: **Docker** (custom), Hardware: **CPU basic** (free is fine).
2. Push this repo there (or point the Space at your GitHub repo and enable
   auto-rebuild on push).
3. Enable **Persistent storage** and set `HF_HOME=/data` (optional but makes
   restarts reuse the downloaded model instead of re-downloading).
4. **Settings → Variables and secrets**:

   | Variable | Value | Why |
   |---|---|---|
   | `MODEL_ID` | `you/spine-severity` | the model repo id (after publishing) |
   | `HF_TOKEN` | `hf_...` | secret, needed for private model repos |
   | `GEMINI_API_KEY` | `AIza...` | optional; enables the LLM report |
   | `API_KEYS` | `key1,key2` | optional; protects the endpoint |
   | `ALLOW_MOCK_FALLBACK` | *(unset)* | demo-without-weights only |

   Production defaults are safe: without weights and without
   `ALLOW_MOCK_FALLBACK=true`, `/api/analyze` returns **503** instead of
   inventing results.
5. Open the Space. Your live URL is
   `https://<you>-spine-severity.hf.space` — that is the URL your frontend
   calls. Visit `/docs` for a Swagger UI of both endpoints.

## Option B — Render

1. New **Web Service** → connect the repo → **Docker** runtime (the
   `Dockerfile` is auto-detected).
2. Environment variables: same set as above (`MODEL_ID`, `HF_TOKEN`, ...).
   Render sets `PORT` itself — the Dockerfile honours it.
3. Deploy. Render gives you `https://<service>.onrender.com` — the live link
   for the frontend. A health-check hitting `/` keeps the free instance awake.

> Tip (any host): the images used for the demo are small sagittal lumbar MRIs
> (single JPEG/PNG per request). CPU inference of ConvNeXt-tiny at 256px is
> well under a second.

---

## API contract (what the frontend calls)

### `GET /`
Service metadata (name, version, endpoints, model status, registry info).

### `GET /api/health`
```json
{
  "status": "ok",
  "version": "2.0.0",
  "vision": {
    "loaded": true, "checkpoint": "...", "model_version": "20260713-091512",
    "image_size": 512, "error": ""
  },
  "registry": {...},
  "mock_fallback_allowed": false,
  "gemini_configured": true
}
```

### `POST /api/analyze`  (multipart/form-data)
Fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | image | required | sagittal lumbar MRI (JPEG/PNG) |
| `use_llm` | string | `"true"` | `"true"/"false"` — LLM report for the case |
| `provider` | string | `""` | e.g. `"gemini"` |
| `case_id` | string | `""` | optional case label |
| `clinical` | string | `"{}"` | JSON `{age, sex, pain_score, symptoms:{...}}` |
| `spondy_predictions` | string | `"[]"` | JSON list of spondylolisthesis inputs |
| `mock_vision` | string | `""` | `"true"` only forces the labelled demo path |

Success `200` returns the `AnalyzeResponse` JSON with:
- the full decision-support case + report,
- `evidence` (per-level Pfirrmann grades with class probabilities),
- `annotated_image_data_url` (data-URL PNG overlay — render it directly in an
  `<img>`),
- `model_status` (which checkpoint/version produced this result),
- `mock_vision: false`.

Errors:
- `400` malformed image or invalid JSON field,
- `401` missing/invalid API key (only when `API_KEYS` is configured),
- `503` no vision model available and mock not requested.

### Frontend call (no SDK needed)
```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);
fd.append("use_llm", "false");
fd.append("clinical", JSON.stringify({ age: 52, symptoms: { back_pain: true } }));

const res = await fetch("https://<your-host>/api/analyze", {
  method: "POST",
  body: fd,
  headers: await (key ? { "X-Api-Key": key } : {}),
});
const data = await res.json();
document.querySelector("#overlay").src = data.annotated_image_data_url;
console.log(data.case.overall_status, data.evidence.ddd);
```
A working single page is in `examples/frontend_demo.html`.

---

## Publishing a (retrained) model

```bash
export HF_TOKEN=hf_...                      # or huggingface-cli login
python publish_model.py --repo acme/spine-severity --with-calibration
```
Then either restart the Space/Render service (picks up latest `main`), or use
versions:
```bash
python publish_model.py --repo acme/spine-severity --rev v2 --with-calibration
# host env: MODEL_ID=acme/spine-severity  HF_MODEL_REVISION=v2
```

The `/api/health` + `/` responses show the active `model_version`, so you can
verify the exact weights being served at all times.

## Security notes

- Set `API_KEYS` if the endpoint is public; frontends must send the key in
  `X-Api-Key` or `Authorization: Bearer`.
- `GEMINI_API_KEY` lives only on the host (never in the repo / frontend).
- Model repos on the Hub: make them private until finished, or gate access,
  and only pull IDs you control.
- CORS default `*` suits a public app; set `CORS_ORIGINS` to your frontend
  origin to restrict it.
- The service never silently produces fake results: the mock path requires an
  explicit `mock_vision=true` (or `ALLOW_MOCK_FALLBACK=true` with no model).