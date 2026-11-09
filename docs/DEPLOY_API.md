# Deploying the Spine API (Hugging Face Spaces / Render)

This guide turns `api_server.py` into a live, public API that a frontend can
call over HTTPS. The key design goal: **you can retrain / improve the model
and update the API without a code change** — weights are versioned in a model
registry and each checkpoint carries its own training metadata.

## Quickstart — run the API locally

From the repo root (Windows PowerShell shown; the commands are the same on
macOS/Linux with `source .venv/bin/activate`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# CPU-only torch keeps the install small; skip this on a CUDA machine
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# needs checkpoints/best_model.pth, or MODEL_ID pointing at a published model
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Open:

- <http://127.0.0.1:8000/ui> — bundled frontend
- <http://127.0.0.1:8000/docs> — Swagger UI / OpenAPI
- <http://127.0.0.1:8000/api/health> — model + registry status

```powershell
# health and a real analysis
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/analyze `
     -F "file=@scan.jpg" -F "explain=true" -F "use_llm=false"
```

If no checkpoint exists locally and `MODEL_ID` is unset, `/api/analyze`
returns **503** by design — it never invents results.

## How it stays working after retraining

1. `train.py` saves a `meta` block in every checkpoint
   (`image_size`, `model_name`, `fpn_dim`, `softargmax_temperature`,
   `tasks`, `version`, ...).
2. `utils.load_model` reads that meta and applies it at load time, so a
   model trained at 512px (or a `convnext_femto` model trained at 160px)
   loads and runs correctly even though the config default differs — no
   config edit needed for the image pipeline to match the checkpoint.
3. `weights.resolve_checkpoint()` picks the active weights from
   (highest priority first):
   - `VISION_CHECKPOINT` env var — a local path **or** an HF Hub repo id
   - `MODEL_ID` env var — an HF Hub repo id (e.g. `acme/spine-severity`)
   - `checkpoints/best_model.pth` (trained locally)
   - `checkpoints/release_model.pth` (committed fp16 artifact)
4. Retraining loop:

   ```
   train on Colab/CPU  ->  checkpoints/best_model.pth
   python publish_model.py --repo acme/spine-severity --with-calibration
   (restart the host; the API downloads the new weights and serves them)
   ```

   For a pinned, reproducible deploy pass `--rev v2` and set
   `HF_MODEL_REVISION=v2` on the host.

> **Weights are not in Git or the Docker image.** `checkpoints/*.pth` is
> excluded by both `.gitignore` and `.dockerignore`. Publish once with
> `publish_model.py` and set `MODEL_ID`, or — for a private image — remove
> `checkpoints/*.pth` from `.dockerignore`, run `python export_release.py`,
> and let the image carry `checkpoints/release_model.pth`.

## One-file pick

- `api_server.py` — FastAPI app (`/`, `/api/health`, `/api/analyze`, `/ui`, `/docs`)
- `examples/frontend_demo.html` — single-page frontend served at `/ui`
- `Dockerfile` — CPU-torch image (works on HF Spaces **and** Render)
- `weights.py` — HF Hub model registry + auto-download
- `publish_model.py` — upload retrained weights to the Hub

---

## Option A — Hugging Face Spaces (recommended)

1. Create a Space at <https://huggingface.co/spaces>:
   - SDK: **Docker** (custom), Hardware: **CPU basic** (free is fine).
2. Keep the Space metadata at the top of `README.md`. The Space UI writes it
   automatically; if you push this repository directly, merge this block into
   the top of `README.md` (otherwise the Space loses its Docker/port config):

   ```yaml
   ---
   title: Spine MRI Decision Support API
   emoji: "\U0001F9B4"
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push this repo there (or point the Space at your GitHub repo and enable
   auto-rebuild on push).
4. Enable **Persistent storage** and set `HF_HOME=/data` (optional but makes
   restarts reuse the downloaded model instead of re-downloading).
5. **Settings → Variables and secrets**:

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
6. Open the Space. Your live URL is
   `https://<you>-spine-severity.hf.space` — that is the URL your frontend
   calls. Visit `/ui` for the bundled frontend and `/docs` for a Swagger UI of
   all endpoints.

## Option B — Render

1. New **Web Service** → connect the repo → **Docker** runtime (the
   `Dockerfile` is auto-detected).
2. Environment variables: same set as above (`MODEL_ID`, `HF_TOKEN`, ...).
   Render sets `PORT` itself — the Dockerfile honours it.
3. Deploy. Render gives you `https://<service>.onrender.com` — the live link
   for the frontend. A health-check hitting `/` keeps the free instance awake.

> Tip (any host): the images used for the demo are small sagittal lumbar MRIs
> (single JPEG/PNG per request). CPU inference of the current
> `convnext_femto` checkpoint at 256 px is roughly 1–3 s per request on the
> free tiers; a `convnext_tiny` model at 512 px needs a GPU to stay snappy.

## Option C — any Docker host / VPS

```bash
docker build -t spine-api .
docker run --rm -p 8000:8000 \
  -e MODEL_ID=you/spine-severity \
  -e API_KEYS=your-key \
  spine-api
```

The container reads `$PORT` (Render) and falls back to 7860 (HF Spaces); map
whichever port you use. Put it behind Caddy/Nginx for HTTPS, or use the
platform URLs from Options A/B.

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
    "loaded": true, "checkpoint": "...", "model_version": "20261107-222501",
    "image_size": 256,
    "tasks": {"localization": true, "ddd": false, "seg": false},
    "ddd_available": false,
    "detector": {"loaded": false, "error": "no weights"},
    "segmentation": {"loaded": false, "error": "no weights"},
    "error": ""
  },
  "registry": {...},
  "mock_fallback_allowed": false,
  "gemini_configured": true
}
```

### `GET /ui`
Serves the bundled frontend (`examples/frontend_demo.html`): upload, annotated
view, Grad-CAM view, per-level table and detection table.

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
| `explain` | string | `"true"` | Grad-CAM overlay (grade CAM with a trained DDD head, else a labelled localization CAM) |
| `mock_vision` | string | `""` | `"true"` only forces the labelled demo path |

Success `200` returns the `AnalyzeResponse` JSON with:
- the full decision-support case + report,
- `evidence` (per-level Pfirrmann grades when available, landmark points,
  geometric indicators, `ddd_available`, `model_tasks`),
- `annotated_image_data_url` (data-URL PNG overlay — render it directly in an
  `<img>`),
- `gradcam_image_data_url` (data-URL PNG — a grade CAM when the DDD head is
  trained, otherwise a `kind="localization"` landmark-attention CAM),
- `detections` (list of boxes with `label`, `class_name`, `box`,
  `confidence`, `source`) and `detection_source`
  (`yolo` → `segmentation` → `landmark` → `none`),
- `segmentation` (`{available, error, ...}`; masks are never invented),
- `gradcam` (`{available, ...}`),
- `ddd_available` (false whenever Pfirrmann labels were absent at training),
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
fd.append("explain", "true");
fd.append("clinical", JSON.stringify({ age: 52, symptoms: { back_pain: true } }));

const res = await fetch("https://<your-host>/api/analyze", {
  method: "POST",
  body: fd,
  headers: await (key ? { "X-Api-Key": key } : {}),
});
const data = await res.json();
document.querySelector("#overlay").src = data.annotated_image_data_url;
document.querySelector("#gradcam").src = data.gradcam_image_data_url;
console.log(data.case.overall_status, data.detections, data.ddd_available);
```
A working single page is in `examples/frontend_demo.html` and is served at
`/ui` by the API itself.

### Run the whole stack locally (end-to-end check)

```powershell
# 1. train / evaluate the localizer (256 px is the current checkpoint)
python train.py --epochs 18 --image-size 256 --batch-size 8 --model convnext_femto --lr 2e-4
python evaluate.py --checkpoint checkpoints/best_model.pth
python preview_predictions.py --num 4      # predicted vs GT overlay PNGs

# 2. start the API
uvicorn api_server:app --host 127.0.0.1 --port 8000

# 3. open the frontend and Swagger UI
#    http://127.0.0.1:8000/ui
#    http://127.0.0.1:8000/docs

# 4. or call it directly
curl -X POST http://127.0.0.1:8000/api/analyze -F "file=@scan.jpg" -F "explain=true"
```

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