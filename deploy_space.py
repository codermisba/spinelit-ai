"""
deploy_space.py
===============

Deploy the API as a Docker Space on the Hugging Face Hub.

The Space image is built from this repository, but the trained weights are
NOT in Git (``checkpoints/*.pth`` / ``*.pkl`` are gitignored).  They are
uploaded separately with ``publish_model.py`` and downloaded at container
start from ``MODEL_ID`` using ``HF_TOKEN``.  This script wires all of that:

    1. creates (or reuses) the Space repo with ``sdk: docker``
    2. uploads the source tree (excluding local data / weights / venv)
    3. writes a Space README with the required Docker frontmatter
    4. sets ``MODEL_ID`` (variable) and ``HF_TOKEN`` (secret) on the Space

Prerequisites
-------------
    python publish_model.py --repo <user>/spine-severity \
        --with-calibration --with-segmentation

Usage
-----
    HF_TOKEN=hf_... python deploy_space.py --model-repo <user>/spine-severity
    python deploy_space.py --space <user>/spine-mri-api \
        --model-repo <user>/spine-severity --public

After the build finishes, the API is at
``https://<user>-<space-name>.hf.space`` (``/ui`` for the frontend).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SPACE_README = """---
title: Spine MRI API
emoji: "🧠"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Spine MRI — localization, segmentation & DDD API

FastAPI service for midsagittal T2 lumbar MRI:

- vertebra/disc localization (ConvNeXt + FPN, soft-argmax)
- standalone U-Net segmentation (vertebra / disc / canal overlay)
- per-level Pfirrmann grading (DDD head) with calibrated P(DDD)
- Grad-CAM explanations and a validated decision-support report

Open `/ui` for the frontend (upload, overlays, per-level table and a real
held-out segmentation gallery), or `/docs` for the OpenAPI UI.

Weights are pulled at startup from the private model repo in `MODEL_ID`
using the `HF_TOKEN` secret.  Research prototype — not a medical device.
"""

# Never upload local-only artefacts (mirrors .dockerignore).
IGNORE = [
    ".venv/*", ".git/*", ".github/*", "git/*",
    "dataset/*", "checkpoints/*", "outputs/*", "logs/*",
    "tests/*", "docs/*", "__pycache__/*", "**/__pycache__/*", "*.pyc",
    ".env", "*.ipynb", "Thumbs.db", ".DS_Store",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy the spine API as a Docker Space.")
    parser.add_argument("--space", default=None,
                        help="Space repo id (default: <user>/spine-mri-api)")
    parser.add_argument("--model-repo", required=True,
                        help="HF model repo with the published weights")
    parser.add_argument("--public", action="store_true",
                        help="make the Space public (default: private)")
    parser.add_argument("--source", default=str(Path(__file__).resolve().parent),
                        help="folder to upload (default: this repository)")
    parser.add_argument("--no-secret", action="store_true",
                        help="do not push HF_TOKEN as a Space secret "
                             "(use when the model repo is public)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("HF_TOKEN must be set (Space creation + private model access).",
              file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    space_id = args.space or f"{user}/spine-mri-api"

    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                    private=not args.public, exist_ok=True)
    print(f"Space ready: {space_id} ({'public' if args.public else 'private'})")

    source = Path(args.source)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(source),
        ignore_patterns=IGNORE,
        commit_message="deploy spine MRI API (localization + U-Net + DDD)",
    )
    print("Uploaded source tree")

    readme = source / ".space_readme.md"
    try:
        readme.write_text(SPACE_README, encoding="utf-8")
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=space_id, repo_type="space",
                        commit_message="Space frontmatter (docker, port 7860)")
    finally:
        readme.unlink(missing_ok=True)
    print("Wrote Space README (sdk: docker, app_port: 7860)")

    try:
        api.add_space_variable(repo_id=space_id, key="MODEL_ID",
                               value=args.model_repo)
        print(f"Variable MODEL_ID={args.model_repo}")
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"Could not set MODEL_ID variable: {exc}", file=sys.stderr)

    if not args.no_secret:
        try:
            api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=token)
            print("Secret HF_TOKEN set (private weights download)")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not set HF_TOKEN secret: {exc}", file=sys.stderr)

    sub = space_id.split("/")[-1]
    print(f"\nLive URL (after the build): "
          f"https://{user}-{sub}.hf.space")
    print(f"Frontend:                  https://{user}-{sub}.hf.space/ui")
    print(f"Health:                    https://{user}-{sub}.hf.space/api/health")
    print(f"Build logs:                https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
