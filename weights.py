"""
weights.py
==========

Model registry for the hosted API.

How the active imaging checkpoint is chosen (first match wins):

    1. ``VISION_CHECKPOINT`` env var
         - a local path, e.g. ``checkpoints/best_model.pth``, or
         - an Hugging Face Hub model id, e.g. ``acme/spine-severity-v2``
    2. ``MODEL_ID`` env var  ->  Hugging Face Hub model id
    3. ``checkpoints/best_model.pth``   (trained locally)
    4. ``checkpoints/release_model.pth`` (committed fp16 artifact)

Hub model ids are resolved with ``hf_hub_download`` into a local cache dir
so the service can be redeployed without re-downloading when the cache is
persisted (set ``HF_HOME`` to a persistent dir on the host).

Retraining workflow ("changing / training a model for betterment"):
    train on Colab -> `python publish_model.py --repo <id>`  ->  update the
    ``MODEL_ID`` secret on the host (or reuse the same repo and just push a
    new revision).  The API picks the new weights at the next restart; the
    checkpoint's own ``meta`` block keeps the runtime consistent with the
    training setup (image size, feature stage, version).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from config import BEST_MODEL, RELEASE_MODEL

logger = logging.getLogger("spine-weights")

HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_model.pth")


class ModelResolutionError(RuntimeError):
    """Raised when no usable checkpoint can be resolved."""


def _is_hub_id(candidate: str) -> bool:
    """
    True when the string looks like an HF Hub repo id (`org/name`), false
    for anything path-like (absolute, ./.., drive letters, subdirs, or a
    weight file name).  Weight-file names (`*.pth`, `*.pt`, ...) are always
    treated as local paths even though they can look like `a/b`.
    """
    if not candidate:
        return False
    if candidate.lower().endswith((".pth", ".pt", ".safetensors", ".bin")):
        return False
    c = candidate.replace("\\", "/")
    if c.startswith(("/", "./", "../")) or c.startswith(".."):
        return False
    parts = c.split("/")
    if len(parts) != 2:
        return False
    return bool(parts[0].strip()) and bool(parts[1].strip())


def resolve_checkpoint() -> Path:
    """
    Return a local path to the active checkpoint, downloading from the
    Hugging Face Hub first if configured.  Raises ModelResolutionError
    when nothing usable exists.
    """
    candidate = (
        os.environ.get("VISION_CHECKPOINT")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()

    if candidate:
        return _resolve(candidate)

    if BEST_MODEL.exists():
        logger.info("Using local checkpoint: %s", BEST_MODEL)
        return BEST_MODEL

    if RELEASE_MODEL.exists():
        logger.info("Using committed release artifact: %s", RELEASE_MODEL)
        return RELEASE_MODEL

    raise ModelResolutionError(
        "No vision checkpoint configured. Set VISION_CHECKPOINT or MODEL_ID "
        "(HF Hub repo id), or add checkpoints/best_model.pth / "
        "checkpoints/release_model.pth."
    )


def _resolve(candidate: str) -> Path:
    if not _is_hub_id(candidate):
        path = Path(candidate)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise ModelResolutionError(
                f"VISION_CHECKPOINT points to a non-existent file: {path}"
            )
        logger.info("Using local checkpoint: %s", path)
        return path
    return download_from_hub(candidate)


def configured_hub_repo() -> str | None:
    """Return the Hub repo id this deployment resolves weights from, if any."""
    candidate = (
        os.environ.get("VISION_CHECKPOINT")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()
    return candidate if candidate and _is_hub_id(candidate) else None


def download_sidecar(filename: str, repo_id: str | None = None) -> Path | None:
    """Best-effort download of an auxiliary artifact from the weights repo.

    Used by optional components (calibration, segmentation) that live in the
    same Hub repo as the main checkpoint.  Returns ``None`` when no Hub repo
    is configured or the file is absent, so an optional task degrades to
    "unavailable" instead of failing startup.
    """
    repo = repo_id or configured_hub_repo()
    if not repo:
        return None
    try:
        return download_from_hub(repo, filename)
    except ModelResolutionError as exc:
        logger.info("Optional artifact %s unavailable on %s: %s",
                    filename, repo, exc)
        return None


def download_from_hub(repo_id: str, filename: str | None = None) -> Path:
    """Download `<filename>` from an HF Hub model repo into the local cache."""
    filename = filename or HF_MODEL_FILENAME
    cache_dir = Path(os.environ.get(
        "MODEL_CACHE_DIR", os.environ.get("HF_HOME", "checkpoints/model_cache")))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ModelResolutionError(
            "huggingface_hub is not installed (add it to requirements.txt)."
        ) from exc

    token = os.environ.get("HF_TOKEN") or None
    revision = os.environ.get("HF_MODEL_REVISION") or None
    logger.info("Downloading %s from HF Hub repo %s (cache: %s)",
                filename, repo_id, cache_dir)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001 - surface all hub failures
        raise ModelResolutionError(
            f"Failed to download {filename} from {repo_id}: {exc}"
        ) from exc

    path = Path(local_path)
    logger.info("Checkpoint ready at %s", path)
    return path


def cache_config_summary() -> dict:
    """Describe where the service will get its weights (for /api/health)."""
    candidate = (
        os.environ.get("VISION_CHECKPOINT")
        or os.environ.get("MODEL_ID")
        or ""
    ).strip()
    return {
        "configured": bool(candidate) or BEST_MODEL.exists() or RELEASE_MODEL.exists(),
        "source": candidate or ("local" if BEST_MODEL.exists() or RELEASE_MODEL.exists() else "none"),
        "hub_revision": os.environ.get("HF_MODEL_REVISION", "") or None,
        "local_best": str(BEST_MODEL),
        "local_release": str(RELEASE_MODEL),
    }


def wipe_cache() -> None:
    """Delete the HF Hub download cache (used by tests/stale-weight fixes)."""
    cache_dir = Path(os.environ.get("HF_HOME", "checkpoints/model_cache"))
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        print("Resolved checkpoint:", resolve_checkpoint())
    except ModelResolutionError as exc:
        print("No checkpoint:", exc)