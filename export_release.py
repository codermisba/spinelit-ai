"""
export_release.py
=================

Convert a full training checkpoint into a small fp16, model-only
checkpoint (checkpoints/release_model.pth) that fits under GitHub's
100 MB file limit and is committed to the repository.

Usage
-----
python export_release.py                       # use checkpoints/best_model.pth
python export_release.py --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import torch

from config import RELEASE_MODEL


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Export a small fp16 release checkpoint for Git."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the full training checkpoint.",
    )

    args = parser.parse_args()

    source = Path(args.checkpoint)

    if not source.exists():
        print(f"Checkpoint not found: {source}")
        return

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    release = {
        key: value.half().clone()
        for key, value in state_dict.items()
    }

    torch.save({"model_state_dict": release}, RELEASE_MODEL)

    print(f"Release model saved to : {RELEASE_MODEL}")


if __name__ == "__main__":
    main()
