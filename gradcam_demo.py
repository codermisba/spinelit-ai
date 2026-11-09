"""
gradcam_demo.py
===============

Run the XAI stage on a midsagittal T2 MRI and export a presentation-ready
Grad-CAM figure (input | heatmap-overlaid).

Grad-CAM explains the Pfirrmann (DDD) head, so it is only meaningful when the
checkpoint was trained with ``dataset/ddd_labels.csv`` (``meta.tasks.ddd``).
If the checkpoint has no trained DDD head the script refuses to present a
heatmap as a grade explanation unless ``--allow-untrained`` is passed; in that
case the figure is clearly labelled as untreated/architectural only.

Usage
-----
    # use checkpoints/best_model.pth (default when present)
    python gradcam_demo.py --image dataset/data/processed_spider_jpgs/170_t2.jpg

    # explicit checkpoint
    python gradcam_demo.py --image scan.jpg --checkpoint checkpoints/best_model.pth

    # architecture-only demo (random DDD head, clearly labelled)
    python gradcam_demo.py --image scan.jpg --allow-untrained
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config as _cfg
from config import DISC_LEVELS, PFRRMANN_GRADES, ROOT_DIR
from explainability import GradCAM, overlay_heatmap
from utils import get_transform, load_model, norm_to_pixels


def _build_untrained_model():
    """Random-head model, ImageNet backbone if available (architecture demo)."""
    from model import SpineFoundationModel

    _cfg.PRETRAINED = True
    model = SpineFoundationModel()
    model.eval()
    return model, torch.device("cpu"), {}, {"localization": True,
                                             "ddd": False, "seg": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="midsagittal T2 MRI (JPG/PNG)")
    ap.add_argument("--output", default=ROOT_DIR / "outputs" / "demo_gradcam.png")
    ap.add_argument("--checkpoint", default="",
                    help="trained .pth (default: checkpoints/best_model.pth)")
    ap.add_argument("--allow-untrained", action="store_true",
                    help="emit a heatmap even when the DDD head is untrained "
                         "(architecture demo only).")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    pil = Image.open(image_path).convert("RGB")
    fname = image_path.stem

    checkpoint = args.checkpoint or str(
        Path(ROOT_DIR) / "checkpoints" / "best_model.pth"
    )
    ddd_available = False

    if Path(checkpoint).exists():
        model, device, _, meta = load_model(checkpoint)
        tasks = dict(meta.get("tasks") or {})
        ddd_available = bool(tasks.get("ddd", False))
        print(f"Checkpoint     : {checkpoint}")
        print(f"Image size     : {meta.get('image_size', _cfg.IMAGE_SIZE)}")
        print(f"DDD head trained: {ddd_available}")
    else:
        print(f"No checkpoint at {checkpoint}; using architecture-only model.")
        model, device, meta, tasks = _build_untrained_model()

    if not ddd_available and not args.allow_untrained:
        raise SystemExit(
            "This checkpoint has no trained DDD (Pfirrmann) head, so a "
            "Grad-CAM heatmap would not explain a real grade.\n"
            "Train with dataset/ddd_labels.csv first, or re-run with "
            "--allow-untrained for an architecture-only demonstration."
        )

    transform = get_transform()
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)
        proba = torch.softmax(preds["ddd_logits"].float(), dim=-1)[0].cpu().numpy()
        disc_px = norm_to_pixels(
            preds["coords"][0][10:20].view(-1, 2).cpu().numpy(),
            *pil.size,
        )

    gcam = GradCAM(model)
    try:
        cams = {d: gcam.cam_for(x, d, int(proba[d].argmax())) for d in range(5)}
    finally:
        gcam.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2), facecolor="white")
    axs[0].imshow(np.asarray(pil.convert("L")), cmap="gray")
    axs[0].set_title("Input: midsagittal T2 MRI", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    overlay = overlay_heatmap(pil, np.mean(list(cams.values()), axis=0),
                              alpha=0.55)
    axs[1].imshow(np.asarray(overlay))
    for d in range(5):
        cls = int(proba[d].argmax())
        px, py = disc_px[d]
        axs[1].scatter(px, py, s=80, marker="D", facecolors="none",
                       edgecolors="white", linewidths=1.5, zorder=4)
        axs[1].annotate(
            f"{DISC_LEVELS[d]} (Pfr {PFRRMANN_GRADES[cls]})",
            (px, py), xytext=(10, -4), textcoords="offset points",
            fontsize=8, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
        )
    title = "Grad-CAM (FPN features used by the DDD head)"
    if not ddd_available:
        title += " — UNTRAINED HEAD, architecture demo only"
    axs[1].set_title(title, fontsize=11, fontweight="bold")
    axs[1].axis("off")

    fig.suptitle(f"Explainable AI — where the spine model looks (case {fname})",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")
    if ddd_available:
        print("predicted Pfirrmann:", [int(proba[d].argmax()) + 1
                                       for d in range(5)])
    else:
        print("WARNING: DDD head is untrained; grades above are random.")


if __name__ == "__main__":
    main()
