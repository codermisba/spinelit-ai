"""
export_release.py
=================

Convert a full training checkpoint into a small fp16, model-only
artifact (checkpoints/release_model.pth) that is easy to store on
Google Drive or share. It is NOT committed to Git any more.

Usage
-----
python export_release.py                                # from best_model.pth
python export_release.py --checkpoint checkpoints/last_model.pth \
                         --output /content/drive/MyDrive/spine/release_model.pth
"""

import argparse
from pathlib import Path

import torch


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Export a portable fp16 inference checkpoint."
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pth",
        help="Full training checkpoint.",
    )
    parser.add_argument(
        "--output", default="checkpoints/release_model.pth",
        help="Where to save the fp16 artifact.",
    )
    args = parser.parse_args()

    source = Path(args.checkpoint)

    if not source.exists():
        print(f"Checkpoint not found: {source}")
        return

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    release = {
        "model_state_dict": {
            key: value.half().clone() for key, value in state_dict.items()
        },
        "tasks": checkpoint.get("tasks", {}),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(release, output)

    print(f"Portable fp16 model saved to : {output}")


if __name__ == "__main__":
    main()
