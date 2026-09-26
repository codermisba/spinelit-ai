"""
publish_model.py
================

Push a trained imaging checkpoint to the Hugging Face Hub so the hosted
API can pick it up by `MODEL_ID` (with zero host-side config changes).

Retraining flow (the "train a better model" path):
    1. train on Colab -> checkpoints/best_model.pth  (and calibration.pkl)
    2. python publish_model.py --repo acme/spine-severity
    3. set MODEL_ID=acme/spine-severity on the host (once)
    4. next publish (with --update) uploads the newer weights; restart the
       host (or bump HF_MODEL_REVISION for a pinned, reproducible deploy).

Usage
-----
    python publish_model.py --repo acme/spine-severity
    python publish_model.py --repo acme/spine-severity --with-calibration
    python publish_model.py --repo acme/spine-severity --rev v2

Uploads overwrite the same filename on the Hub by default; keep a distinct
`--rev` (or distinct filename) when you want version pinning.  Uses the
HF_TOKEN secret (or `huggingface-cli login`).
"""

import argparse
import sys
from pathlib import Path

from config import BEST_MODEL, CALIB_ARTIFACT, RELEASE_MODEL


def upload(repo_id: str, local_path: Path, filename: str,
           revision: str | None, hf_token: str | None) -> str:
    from huggingface_hub import upload_file

    url = upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        token=hf_token,
        commit_message=f"publish {filename} ({local_path.stat().st_size} bytes)",
        revision=revision or "main",
    )
    return url


def main() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Publish a trained imaging checkpoint to the HF Hub.")
    parser.add_argument("--repo", required=True,
                        help="HF Hub model repo id, e.g. acme/spine-severity")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL),
                        help="checkpoint to upload (name kept as-is)")
    parser.add_argument("--filename", default=None,
                        help="uploaded filename (default: checkpoint basename)")
    parser.add_argument("--rev", default=None,
                        help="target revision (default: main; pass v2/v3 for pinned deploys)")
    parser.add_argument("--with-calibration", action="store_true",
                        help="also upload calibration.pkl if it exists")
    args = parser.parse_args()

    source = Path(args.checkpoint)
    if not source.exists():
        print(f"Checkpoint not found: {source}", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("HF_TOKEN is not set. Uploads to private/repo-gated models "
              "will fail; public repos may still work from an authenticated "
              "`huggingface-cli login`.", file=sys.stderr)

    files = [(source, args.filename or source.name)]
    if args.with_calibration and CALIB_ARTIFACT.exists():
        files.append((CALIB_ARTIFACT, CALIB_ARTIFACT.name))

    for path, remote_name in files:
        try:
            url = upload(args.repo, path, remote_name, args.rev, token)
            print(f"Uploaded {path.name} -> {url}")
        except Exception as exc:  # noqa: BLE001 - friendly CLI errors
            print(f"Upload failed for {path.name}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"\nDone. Point the hosted API at it:")
    print(f'  set MODEL_ID={args.repo}' + (f'  (pin HF_MODEL_REVISION={args.rev})' if args.rev else '')
          + '\n  then restart the hosted API to serve the new weights.')


if __name__ == "__main__":
    main()