"""
plot_overlay.py
===============

Paper-style point-overlay plots for the SPIDER DDD (Pfirrmann) pipeline.

For a single midsagittal T2 case it draws a publication-quality figure:
  * Left panel   — the raw T2 image (greyscale).
  * Right panel  — the same image overlaid with per-disc markers coloured by
                   predicted Pfirrmann grade, each labelled with its level and
                   a Pfirrmann numeral, plus locating guide-lines.
  * A horizontal colourbar legend mapping Pfirrmann grade -> colour.

This mirrors the overlay style used in the 2026 systematic review figures
(AUROC 0.86-0.99 / accuracy up to 97% DDD classification) — clean,
labelled, colour-coded discs on the anatomical image.

Usage
-----
python plot_overlay.py --image dataset/data/processed_spider_jpgs/case_midsag.jpg \
    --coords dataset/coords_pretrain.csv \
    --grades dataset/ddd_labels.csv \
    --output outputs/overlay_case.png \
    --title "Lumbar Spine | Pfirrmann DDD Grading"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd
from PIL import Image

from config import DISC_LEVELS, PFRRMANN_GRADES, ROOT_DIR

# Pfirrmann grade -> colour (qualitative, colourblind-friendly).
GRADE_CMAP = {
    1: "#2E86AB",   # I   (normal)        blue
    2: "#43AA8B",   # II                  teal/green
    3: "#F9C74F",   # III                 amber
    4: "#F9844A",   # IV                  orange
    5: "#F94144",   # V                   red
}
GRADE_COLORS = [GRADE_CMAP[g] for g in range(1, 6)]


def load_coords(coords_csv: Path, filename: str) -> dict:
    df = pd.read_csv(coords_csv)
    df = df[df["filename"].astype(str).str.strip() == filename]
    out = {}
    for _, row in df.iterrows():
        out[str(row["level"]).strip()] = (
            float(row["relative_x"]), float(row["relative_y"])
        )
    return out


def load_grades(grades_csv: Path, filename: str) -> dict:
    if not Path(grades_csv).exists():
        return {}
    df = pd.read_csv(grades_csv)
    df = df[df["filename"].astype(str).str.strip() == filename]
    out = {}
    for _, row in df.iterrows():
        level = str(row["level"]).strip()
        try:
            out[level] = int(float(row["pfirrmann_grade"]))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def draw_colorbar(ax, title: str) -> None:
    """Horizontal legend: each Pfirrmann grade as a labelled colour swatch."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    n = len(GRADE_CMAP)
    for i, g in enumerate(range(1, n + 1)):
        x0 = i / n
        ax.add_patch(plt.Rectangle((x0, 0.05), 1 / n, 0.9,
                                   facecolor=GRADE_CMAP[g], edgecolor="none"))
        ax.text(x0 + 0.5 / n, 0.05 - 0.12, f"{PFRRMANN_GRADES[g-1]}",
                ha="center", va="top", fontsize=9, color="white",
                weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def plot_case(
    image_path: Path,
    coords: dict,
    grades: dict,
    output_path: Path,
    title: str,
    dpi: int = 300,
) -> None:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
    h, w = arr.shape

    fig, (ax_img, ax_overlay) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor="white"
    )

    for ax, overlay in ((ax_img, False), (ax_overlay, True)):
        ax.imshow(arr, cmap="gray", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not overlay:
            ax.set_title("Midsagittal T2", fontsize=12, fontweight="bold")

    levels = [lv for lv in DISC_LEVELS if lv in coords]

    for level in levels:
        rx, ry = coords[level]
        px, py = rx * w, ry * h
        grade = grades.get(level)
        color = GRADE_CMAP.get(grade, "#CCCCCC")
        label = level
        if grade is not None:
            numeral = PFRRMANN_GRADES[grade - 1]
            label = f"{level}\nPfr {numeral}"

        ax_overlay.scatter(
            px, py, s=340, marker="o", facecolors="none",
            edgecolors=color, linewidths=2.5, zorder=3
        )
        ax_overlay.scatter(px, py, s=90, c=color, marker="o", zorder=4)
        ax_overlay.annotate(
            label, (px, py), textcoords="offset points", xytext=(14, 6),
            fontsize=9, color="#111111", weight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                      alpha=0.75),
        )

    ax_overlay.set_title("Pfirrmann DDD Overlay", fontsize=12,
                         fontweight="bold")

    # Colourbar legend spanning the bottom.
    ax_cb = fig.add_axes([0.12, 0.05, 0.76, 0.05])
    draw_colorbar(ax_cb, "Pfirrmann grade")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved overlay -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--coords", default=ROOT_DIR / "dataset" / "coords_pretrain.csv")
    ap.add_argument("--grades", default=ROOT_DIR / "dataset" / "ddd_labels.csv")
    ap.add_argument("--output", default=ROOT_DIR / "outputs" / "overlay_case.png")
    ap.add_argument("--title", default="Lumbar Spine | Pfirrmann DDD Grading")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    image_path = Path(args.image)
    filename = image_path.name

    coords = load_coords(Path(args.coords), filename)
    grades = load_grades(Path(args.grades), filename)

    if not coords:
        raise SystemExit(f"No landmarks found for {filename} in {args.coords}")

    plot_case(image_path, coords, grades, Path(args.output), args.title,
              args.dpi)


if __name__ == "__main__":
    main()
