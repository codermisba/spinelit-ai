"""
gradcam_demo.py
===============

Run the SpineLIT XAI stage on a midsagittal T2 MRI and export a
presentation-ready Grad-CAM figure (input | heatmap-overlaid).

Uses the severity (ConvNeXt) model's own predictions, so no label file is
needed. On a CPU-only machine this is slow but works (one 512x512 image,
one backward pass).

Usage
-----
    python gradcam_demo.py --image dataset/data/processed_spider_jpgs/170_t2.jpg
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

from config import DISC_LEVELS, ROOT_DIR
from explainability import GradCAM, overlay_heatmap
from model import SpineFoundationModel
from utils import draw_landmarks, norm_to_pixels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="midsagittal T2 MRI (JPG/PNG)")
    ap.add_argument("--output", default=ROOT_DIR / "outputs" / "demo_gradcam.png")
    ap.add_argument("--checkpoint", default="", help="optional fine-tuned .pth")
    args = ap.parse_args()

    from torchvision import transforms

    image_path = Path(args.image)
    pil = Image.open(image_path).convert("RGB")
    fname = image_path.stem

    model = SpineFoundationModel()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = transform(pil).unsqueeze(0)

    with torch.no_grad():
        preds = model(x)
        proba = torch.softmax(preds["ddd_logits"], dim=-1)[0].cpu().numpy()
        disc_px = norm_to_pixels(
            preds["coords"][0][10:20].view(-1, 2).cpu().numpy(),
            *pil.size,
        )

    gcam = GradCAM(model)
    cams = {}
    for d in range(5):
        cls = int(proba[d].argmax())
        cams[d] = gcam.cam_for(x, d, cls)
    gcam.close()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2), facecolor="white")
    axs[0].imshow(np.asarray(pil.convert("L")), cmap="gray")
    axs[0].set_title("Input: midsagittal T2 MRI", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    overlay = overlay_heatmap(pil, np.mean(list(cams.values()), axis=0), alpha=0.55)
    axs[1].imshow(np.asarray(overlay))
    for d, cls in enumerate([int(proba[d].argmax()) for d in range(5)]):
        px, py = disc_px[d]
        axs[1].scatter(px, py, s=80, marker="D", facecolors="none",
                       edgecolors="white", linewidths=1.5, zorder=4)
        axs[1].annotate(f"{DISC_LEVELS[d]} (Pfr {['I','II','III','IV','V'][cls]})",
                        (px, py), xytext=(10, -4), textcoords="offset points",
                        fontsize=8, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55))
    axs[1].set_title("Grad-CAM (severity network)", fontsize=12, fontweight="bold")
    axs[1].axis("off")

    fig.suptitle(f"Explainable AI — where the spine model looks (case {fname})",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")
    print("predicted Pfirrmann:", [int(proba[d].argmax()) + 1 for d in range(5)])


if __name__ == "__main__":
    main()