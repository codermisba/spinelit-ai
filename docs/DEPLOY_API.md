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
   python publish_model.py --repo acme/spine-severity --with-calibration --with-segmentation
   (restart the host; the API downloads the new weights and serves them)
   ```

   `--with-segmentation` uploads `checkpoints/segmentation.pth` so the hosted
   API serves real masks too; without it the API honestly reports
   `segmentation.available=false` and falls back to landmark boxes.

   For a pinned, reproducible deploy pass `--rev v2` and set
   `HF_MODEL_REVISION=v2` on the host.

> **Weights are not in Git or the Docker image.** `checkpoints/*.pth` is
> excluded by both `.gitignore` and `.dockerignore`. Publish once with
> `publish_model.py` and set `MODEL_ID`, or — for a private image — remove
> `checkpoints/*.pth` from `.dockerignore`, run `python export_release.py`,
> and let the image carry `checkpoints/release_model.pth`.

## One-file pick

- `api_server.py` — FastAPI app (`/`, `/api/health`, `/api/analyze`, `/ui`,
  `/integration`, `/docs`, `/api/segmentation/samples`)
- `examples/frontend_demo.html` — single-page frontend served at `/ui`
- `docs/FRONTEND_INTEGRATION.md` — integration guide served at `/integration`
- `explanation.py` — grounded LLM/deterministic explanation of the model output
- `Dockerfile` — CPU-torch image (works on HF Spaces **and** Render)
- `weights.py` — HF Hub model registry + auto-download
- `publish_model.py` — upload retrained weights to the Hub

---

## Option A — Hugging Face Spaces

> **Note (2026):** Hugging Face now requires a **PRO** subscription to host a
> Docker (or Gradio) Space on `cpu-basic`; the API returns
> `402 Payment Required` on Space creation without it. Static Spaces are still
> free but cannot run this service. Use Option B (Render) for a free host, or
> upgrade the account and re-run `deploy_space.py`.

### Automated: one command

Once the weights are on the Hub (`publish_model.py`, see below), the Space can
be created, configured and uploaded in a single step:

```bash
export HF_TOKEN=hf_...            # PowerShell: $env:HF_TOKEN="hf_..."
python publish_model.py --repo you/spine-severity \
    --with-calibration --with-segmentation
python deploy_space.py --model-repo you/spine-severity
```

`deploy_space.py` creates the Docker Space (private by default, `--public` to
open it), uploads the source tree, writes the README frontmatter below and
sets `MODEL_ID` (variable) plus `HF_TOKEN` (secret) so the container can pull
the private weights at startup. It prints the live `/ui`, `/api/health` and
build-log URLs. Re-run it after any code change to redeploy.

### Manual

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
   | `GEMINI_API_KEY` | `AIza...` | optional; enables the LLM report and explanation |
   | `API_KEYS` | `key1,key2` | optional; protects the endpoint |
   | `ALLOW_MOCK_FALLBACK` | *(unset)* | demo-without-weights only |

   Production defaults are safe: without weights and without
   `ALLOW_MOCK_FALLBACK=true`, `/api/analyze` returns **503** instead of
   inventing results.
6. Open the Space. Your live URL is
   `https://<you>-spine-severity.hf.space` — that is the URL your frontend
   calls. Visit `/ui` for the bundled frontend and `/docs` for a Swagger UI of
   all endpoints.

## Option B — Render (free, blueprint included)

`render.yaml` describes the whole service, so the deploy is point-and-click:

1. Push this repo to GitHub (already: `codermisba/spinelit-ai`).
2. Render Dashboard → **New → Blueprint** → select the repo → **Apply**.
3. When Render asks for `HF_TOKEN`, paste the Hub token (the model repo is
   private). `MODEL_ID` is already set to the published repo.
4. The live URL is `https://<service>.onrender.com`; open `<url>/ui`.

Constraints on the free instance (512 MB RAM, 0.1 CPU): the container sets
`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`, inference takes tens of seconds, and
the instance sleeps after ~15 min idle (first request after that is a cold
start that also re-downloads ~107 MB of weights). Upgrade to the Starter plan
for 512 MB→2 GB and real CPU if you want a responsive demo.

Manual alternative: New **Web Service** → connect the repo → **Docker**
runtime → add the same environment variables (`MODEL_ID`, `HF_TOKEN`, ...).
Render sets `PORT` itself — the Dockerfile honours it. A health check on
`/api/health` keeps the instance honest.

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
    "loaded": true, "checkpoint": "...", "model_version": "20261109-120000",
    "image_size": 256,
    "tasks": {"localization": true, "ddd": true, "seg": false},
    "ddd_available": true,
    "detector": {"loaded": false, "error": "no weights"},
    "segmentation": {"loaded": true, "val_dice": 0.486,
                     "val_dice_per_class": [0.95, 0.58, 0.37, 0.51]},
    "error": ""
  },
  "registry": {...},
  "mock_fallback_allowed": false,
  "gemini_configured": true
}
```

`tasks.seg` refers to the segmentation head **inside** the shared checkpoint.
The API also serves masks from a standalone U-Net (`checkpoints/segmentation.pth`
or `segmentation.pth` in the Hub repo); that is reported separately under
`vision.segmentation`, and either source makes `segmentation.available` true.
All availability fields are derived from actually loaded weights.

### `GET /integration`
Serves `docs/FRONTEND_INTEGRATION.md` as `text/markdown; charset=utf-8` with
`Cache-Control: no-store`. This is the guide to hand to a frontend team
(hosted URL, CORS, fields, response shapes, error codes, JS example,
checklist).

### `GET /ui`
Serves the bundled frontend (`examples/frontend_demo.html`): upload, annotated
view, Grad-CAM view, U-Net overlay, per-level table, detection table and a
segmentation test gallery. The response is sent with `Cache-Control: no-store`
so a redeployed frontend is never masked by a stale browser cache.

### `GET /api/segmentation/samples`
Lists the real held-out segmentation overlays available to the frontend:

```json
{
  "count": 4,
  "dirs": [".../examples/segmentation_samples", ".../outputs/segmentation"],
  "samples": [
    {"name": "100_t2_overlay", "file": "100_t2_overlay.png",
     "source": "samples", "size_bytes": 97248,
     "url": "/api/segmentation/samples/100_t2_overlay.png"}
  ]
}
```

Each PNG is the 3-panel evaluation figure (`MRI | ground truth | prediction`)
written by `evaluate_segmentation.py`. `examples/segmentation_samples/` is
committed, so the gallery works on a hosted deployment even though `outputs/`
is excluded from Git and Docker; locally-produced overlays are added
automatically. Individual files are served at
`GET /api/segmentation/samples/{file}` (whitelisted `.png` names only).

### `POST /api/analyze`  (multipart/form-data)
Fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | image | required | sagittal lumbar MRI (JPEG/PNG) |
| `use_llm` | string | `"true"` | `"true"/"false"` — LLM report **and** explanation (falls back to deterministic text when no key/LLM) |
| `provider` | string | `""` | e.g. `"gemini"` |
| `case_id` | string | `""` | optional case label |
| `clinical` | string | `"{}"` | JSON `{age, sex, pain_score, symptoms:{...}}` |
| `spondy_predictions` | string | `"[]"` | JSON list of spondylolisthesis inputs |
| `explain` | string | `"true"` | Grad-CAM overlay (grade CAM with a trained DDD head, else a labelled localization CAM) |
| `mock_vision` | string | `""` | `"true"` only forces the labelled demo path |

Success `200` returns the `AnalyzeResponse` JSON with:
- the full decision-support case + report,
- `evidence` (per-level Pfirrmann grades when available, landmark points,
  geometric indicators, `ddd_available`, `model_tasks`). Each level carries
  `class_probabilities` (5 softmax values), `raw_confidence` (max class
  probability) and `calibrated_probability` = **calibrated P(Pfirrmann ≥ 2)**,
  i.e. the probability of any degeneration at that disc, not a confidence in
  the exact grade. The calibrator is loaded from `checkpoints/calibration.pkl`;
  when that file is missing on a hosted deployment it is downloaded from the
  `MODEL_ID` repo, and with no artifact at all the raw P(DDD) is served
  unchanged (no heuristic rescaling),
- `annotated_image_data_url` (data-URL PNG overlay — render it directly in an
  `<img>`),
- `input_image_data_url` (data-URL PNG of the uploaded MRI, so the UI can show
  the original beside the segmentation overlay),
- `segmentation_image_data_url` (data-URL PNG: the U-Net class mask blended
  over the uploaded scan; `""` when segmentation is unavailable or on the
  explicit demo path),
- `gradcam_image_data_url` (data-URL PNG — a grade CAM when the DDD head is
  trained, otherwise a `kind="localization"` landmark-attention CAM),
- `detections` (list of boxes with `label`, `class_name`, `box`,
  `confidence`, `source`) and `detection_source`
  (`yolo` → `landmark` → `segmentation` → `none`; landmark boxes keep their
  per-level labels and carry the localizer's per-point confidence),
- `segmentation` (`{available, error, ...}`; masks are never invented),
- `gradcam` (`{available, ...}`),
- `ddd_available` (false whenever Pfirrmann labels were absent at training),
- `explanation` (`{text, source, llm_used, model, error}`; `source` is
  `"gemini"`/provider when the LLM produced the prose, otherwise
  `"deterministic"`. Both modes use only the model evidence and never invent
  findings; when the LLM call fails, the deterministic text is returned with
  the error recorded),
- `model_status` (which checkpoint/version produced this result),
- `mock_vision: false`.

DDD limitation to surface in any frontend: SPIDER has one labelled sagittal
slice per case, so exact Pfirrmann accuracy is modest (about one third of 210
validation discs, majority class ≈ 28%); within-one-grade agreement is ~80%.
The field is reported honestly with per-class probabilities and is **not** a
clinical diagnosis.

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
# 1. localizer (256 px is the current checkpoint)
python train.py --epochs 18 --image-size 256 --batch-size 8 --model convnext_femto --lr 2e-4
python evaluate.py --checkpoint checkpoints/best_model.pth
python preview_predictions.py --num 4      # predicted vs GT overlay PNGs

# 2. DDD head (needs dataset/ddd_labels.csv from prepare_spider.py)
python finetune_ddd.py --epochs 250 --patience 40 --select-by loss

# 3. segmentation + calibration (optional but recommended)
python prepare_segmentation_pairs.py
python train_segmentation.py --images dataset/segmentation/images `
    --masks dataset/segmentation/masks --epochs 40
python evaluate_segmentation.py
python train_calibrator.py

# 4. start the API (loads best_model.pth + segmentation.pth)
uvicorn api_server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/health

# 5. open the frontend and Swagger UI
#    http://127.0.0.1:8000/ui
#    http://127.0.0.1:8000/docs

# 6. or call it directly
curl -X POST http://127.0.0.1:8000/api/analyze -F "file=@scan.jpg" -F "explain=true"
```

---

## Publishing a (retrained) model

```bash
export HF_TOKEN=hf_...                      # or huggingface-cli login
python publish_model.py --repo acme/spine-severity \
    --with-calibration --with-segmentation
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