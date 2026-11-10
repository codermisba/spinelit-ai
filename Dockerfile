# Spine MRI Decision Support API - hosted service
#
# One Docker image for Hugging Face Spaces AND Render (both support Docker).
# Uses the CPU torch build so cold starts are fast and the free tiers work.
#
# Build / run locally:
#   docker build -t spine-api .
#   docker run --rm -p 8000:8000 -e MODEL_ID=your/spine-model spine-api
#
# Hosted (HF Spaces): set "Hardware = CPU basic", Dockerfile = this file,
# Secrets: MODEL_ID / HF_TOKEN / GEMINI_API_KEY / API_KEYS / HF_HOME=/data
# (enable persistent storage so the model download survives restarts).
# Hosted (Render): Web Service -> Docker, env vars as above, PORT handled by
# the platform.

FROM python:3.12-slim

# Keep CPU inference inside small free-tier limits (512 MB / 0.1 CPU on
# Render): one thread, glibc's arena cap, no Xet upload/download path.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch (the default wheel is the ~2GB CUDA build).
# Install separately so `pip install -r requirements.txt` skips it.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Source (excluding local data / weights / venv via .dockerignore).
COPY . .

EXPOSE 7860 8000

# Render injects $PORT; HF Spaces proxies 7860 by default (can be overridden
# with a PORT secret). Fallback 7860 keeps both platforms working out of the box.
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-7860}