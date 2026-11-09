"""
yolo_train.py
=============

Train the optional Ultralytics YOLO vertebra/disc detector.

The detector is **optional**: the API works without it (landmark-derived
blocks are used and honestly labelled).  A trained detector gives tighter,
mask-like boxes and a second independent view of the anatomy.

Workflow
--------
    # 1. build pseudo-boxes from the existing keypoints (no manual labelling)
    python yolo_prep.py --from_landmarks --out_dir dataset/yolo

    # 2. train (CPU is slow but works; GPU/Colab recommended)
    python yolo_train.py --data dataset/yolo/data.yaml --epochs 60

    # 3. the best checkpoint is copied to checkpoints/yolo_best.pt

Alternative: build true boxes from SPIDER masks (if you have the volumes):
    python yolo_prep.py --images_dir ... --masks_dir ... --out_dir dataset/yolo
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config as _cfg
from config import YOLO_CHECKPOINT, YOLO_DATA_DIR


def ensure_dataset(data_yaml: Path, from_landmarks: bool) -> Path:
    if data_yaml.exists():
        return data_yaml
    if from_landmarks:
        from yolo_prep import prepare_from_landmarks
        out_dir = data_yaml.parent
        prepare_from_landmarks(out_dir)
        if data_yaml.exists():
            return data_yaml
    raise SystemExit(
        f"YOLO dataset descriptor not found: {data_yaml}. Build it first:\n"
        "  python yolo_prep.py --from_landmarks --out_dir dataset/yolo"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the optional YOLO vertebra/disc detector."
    )
    ap.add_argument("--data", default=str(YOLO_DATA_DIR / "data.yaml"))
    ap.add_argument("--from-landmarks", action="store_true",
                    help="Auto-build pseudo-boxes if data.yaml is missing.")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Ultralytics base model (yolo11n.pt / yolov8n.pt).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None,
                    help="'cpu', '0' (first GPU), or omit for auto.")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", default="outputs/yolo_runs")
    ap.add_argument("--name", default="spine_detector")
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is not installed. Install it with:\n"
            "  pip install ultralytics"
        ) from exc

    data_yaml = ensure_dataset(Path(args.data), args.from_landmarks)

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        patience=int(args.patience),
        exist_ok=True,
        plots=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    print(f"Training YOLO on {data_yaml} ...")
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        YOLO_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, YOLO_CHECKPOINT)
        print(f"Best detector copied to {YOLO_CHECKPOINT}")
    else:
        print(f"Training finished but best.pt was not found under {save_dir}.")

    metrics = getattr(results, "results_dict", None)
    if metrics:
        print("Metrics:", {k: round(float(v), 4) for k, v in metrics.items()
                          if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
