"""
utils.py
========

Shared helpers for the Spine Foundation Model:

- Preprocessing (must match training)
- Model / checkpoint loading
- Single-image inference + output decoding
- Geometric indicators (disc-space narrowing / listhesis proxy)
- Landmark visualization on the ORIGINAL image dimensions
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from config import (
    BEST_MODEL,
    DDD_LABELS_CSV,
    IMAGE_SIZE,
    LISTHESIS_OFFSET_THRESHOLD,
    NARROWED_DISC_THRESHOLD,
    NUM_KEYPOINTS,
    PFRRMANN_GRADES,
    RELEASE_MODEL,
    VERTEBRA_LANDMARK_CSV,
)
from config import VERTEBRAE, DISC_LEVELS
from model import SpineFoundationModel

# Fixed point order everywhere: 5 vertebral centres then 5 disc centres
ALL_POINTS = list(VERTEBRAE) + list(DISC_LEVELS)

VERTEBRA_COLOURS = [
    (0, 229, 255),    # cyan
    (80, 255, 120),   # green
    (255, 200, 0),    # amber
    (255, 100, 255),  # magenta
    (255, 130, 70),   # orange
]

DISC_COLOURS = [
    (0, 150, 255),    # blue
    (170, 255, 60),   # lime
    (255, 240, 60),   # yellow
    (255, 90, 180),   # pink
    (140, 160, 255),  # periwinkle
]

PREDICTION_COLOUR = (255, 60, 60)
GROUND_TRUTH_COLOUR = (60, 255, 60)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def get_transform() -> transforms.Compose:
    """Return the exact transform used during training."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into the (3,H,W) tensor the model expects."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return get_transform()(image)


# ---------------------------------------------------------
# Optional label availability (for UI status messages)
# ---------------------------------------------------------

def ddd_labels_available() -> bool:
    return Path(DDD_LABELS_CSV).exists()


def vertebra_labels_available() -> bool:
    return Path(VERTEBRA_LANDMARK_CSV).exists()


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------

def load_model(
    checkpoint_path: Union[str, Path] = BEST_MODEL,
    device: Optional[torch.device] = None,
):
    """
    Load the Spine Foundation Model from a training checkpoint.

    Returns
    -------
    (model, device, checkpoint_path)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)

    # Fresh clone without training: fall back to the committed fp16 artifact
    if not checkpoint_path.exists() and RELEASE_MODEL.exists():
        checkpoint_path = RELEASE_MODEL

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained model checkpoint found: {checkpoint_path}"
            f" — train first with: python train.py"
        )

    model = SpineFoundationModel().to(device)
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    first_tensor = next(iter(state_dict.values()))
    if first_tensor.dtype == torch.float16:
        model = model.half()

    model.load_state_dict(state_dict)

    return model, device, checkpoint_path


# ---------------------------------------------------------
# Inference + decoding
# ---------------------------------------------------------

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device) -> dict:
    """Run inference on a single preprocessed image tensor."""
    input_batch = image_tensor.unsqueeze(0).to(device)

    if next(model.parameters()).dtype == torch.float16:
        input_batch = input_batch.half()

    return model(input_batch)


def decode_outputs(outputs: dict) -> dict:
    """
    Convert raw model outputs into interpretable numpy values:

        points             : (10,2) normalized coordinates
        localization_conf  : (10,) 0-1
        ddd_prob           : (5,5) softmax Pfirrmann class probabilities
        ddd_grade          : (5,) predicted Pfirrmann grade (1-5)
        ddd_conf           : (5,) 0-1 maximum class probability per disc
    """
    import numpy as np

    logits = outputs["ddd_logits"].float().detach().cpu().numpy()
    # Squeeze any leading batch dim: predict() returns (1,5,5).
    logits = logits.reshape(-1, logits.shape[-1])          # (5, 5) per disc
    probs = _softmax(logits, axis=-1)

    return {
        "points": outputs["coords"].float().detach().cpu().numpy()
                  .reshape(NUM_KEYPOINTS, 2),
        "localization_conf": outputs["localization_conf"].float().detach().cpu()
                             .numpy().reshape(NUM_KEYPOINTS),
        "ddd_prob": probs,
        "ddd_grade": (np.argmax(probs, axis=-1) + 1.0).reshape(-1),
        "ddd_conf": probs.max(axis=-1).reshape(-1),
    }


def _softmax(x, axis=-1) -> np.ndarray:
    import numpy as np
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def norm_to_pixels(coords, width: int, height: int) -> np.ndarray:
    """Convert normalized (0-1) coordinates to ORIGINAL pixel coordinates."""
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    pixels = np.empty_like(coords)
    pixels[:, 0] = coords[:, 0] * width
    pixels[:, 1] = coords[:, 1] * height
    return pixels


# ---------------------------------------------------------
# Geometric indicators (interpretable, no diagnosis implied)
# ---------------------------------------------------------

def severity_from_grade(pfirrmann_grade: float) -> str:
    """Map a Pfirrmann grade (1-5) to a severity label."""
    g = round(float(pfirrmann_grade))
    if g <= 1:
        return "Normal"
    if g == 2:
        return "Mild"
    if g == 3:
        return "Moderate"
    return "Severe"


def pfirrmann_label(grade: float) -> str:
    g = min(max(int(round(float(grade))) - 1, 0), len(PFRRMANN_GRADES) - 1)
    return PFRRMANN_GRADES[g]


def compute_geometric_indicators(points_px: np.ndarray) -> list[dict]:
    """
    Compute label-free geometric indicators per disc level from the
    predicted vertebral-body centres (sagittal images).

    - relative_space : inter-centre distance vs mean (low -> narrowed space)
    - offset_ratio   : horizontal centre offset vs mean distance
                       (high -> possible antero/retrolisthesis)

    Only levels with adjacent vertebra centres are reported
    (L1/L2 ... L4/L5; L5/S1 needs the sacrum which is not a keypoint).
    """
    vertebra_pts = points_px[:len(VERTEBRAE)]

    distances = [
        float(np.linalg.norm(vertebra_pts[i + 1] - vertebra_pts[i]))
        for i in range(len(VERTEBRAE) - 1)
    ]

    if not distances or np.mean(distances) < 1e-6:
        return []

    mean_distance = float(np.mean(distances))

    indicators = []
    for i in range(len(distances)):
        relative_space = distances[i] / mean_distance
        offset_ratio = (
            abs(float(vertebra_pts[i][0] - vertebra_pts[i + 1][0]))
            / mean_distance
        )
        indicators.append(
            {
                "level": DISC_LEVELS[i],
                "relative_space": round(relative_space, 3),
                "narrowed": relative_space < NARROWED_DISC_THRESHOLD,
                "offset_ratio": round(offset_ratio, 3),
                "offset_flag": offset_ratio > LISTHESIS_OFFSET_THRESHOLD,
            }
        )

    return indicators


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def _to_pil_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert("RGB")


def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---- Thin-line anatomical overlay ---------------------------------
#
# The model predicts 10 keypoints (5 vertebra centres + 5 disc
# centres), NOT a pixel segmentation mask. To give the output image a
# readable "segmented vertebrae" appearance we render a THIN-LINE
# schematic that is derived *only* from the detected keypoints:
#
#   * one thin outline rectangle per vertebra body (bounds = the
#     detected disc centres above/below the vertebra centre),
#   * thin axis lines connecting consecutive vertebra centres,
#   * thin horizontal offset lines = the exact horizontal displacement
#     used by `compute_geometric_indicators` for listhesis detection.
#
# This is a landmark-derived schematic overlay — it is NOT a learned
# segmentation mask and must not be presented as one.
# --------------------------------------------------------------------

ANATOMY_LINE_WIDTH = 1          # thin lines only
DRAW_ANATOMY_THIN_LINES = True

BODY_OUTLINE_COLOUR = (80, 200, 255)    # light blue - vertebra bodies
AXIS_LINE_COLOUR = (130, 230, 130)      # green - spinal column axis
OFFSET_LINE_COLOUR = (255, 195, 60)     # amber - horizontal displacement


def _anatomy_body_rects(vert_px: np.ndarray, disc_px: np.ndarray,
                        mean_distance: float) -> list[list[float]]:
    """
    Derive a thin rectangular outline for each vertebra body from the
    detected vertebra centres and the two adjacent disc centres.
    """
    rects = []
    half_w = 0.5 * mean_distance * 0.92
    n = len(vert_px)
    for i in range(n):
        cx, cy = float(vert_px[i][0]), float(vert_px[i][1])
        if i == 0:
            top_y = cy - (float(disc_px[0][1]) - cy)     # mirror above L1/L2
            bottom_y = float(disc_px[0][1])
        elif i < n - 1:
            top_y = float(disc_px[i - 1][1])
            bottom_y = float(disc_px[i][1])
        else:
            top_y = float(disc_px[i - 1][1])
            bottom_y = cy + (cy - float(disc_px[i - 1][1]))  # mirror below
        half_h = max((bottom_y - top_y) / 2.0, mean_distance * 0.2)
        rects.append([cx - half_w, cy - half_h, cx + half_w, cy + half_h])
    return rects


def draw_thin_anatomy_overlay(
    image,
    coords,
    offset_labels: bool = True,
) -> Image.Image:
    """
    Draw only the thin-line anatomical schematic (vertebra outlines,
    column axis, offset lines) on the ORIGINAL image dimensions.

    `offset_labels=False` skips the per-pair "off 0.xx" text (used by
    the combined landmark view to keep the image uncluttered).

    Derived solely from the 10 detected keypoints; no segmentation-model
    output is implied.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size
    points_px = norm_to_pixels(coords, width, height).astype(np.float64)

    vertebra_px = points_px[:len(VERTEBRAE)]
    disc_px = points_px[len(VERTEBRAE):]

    distances = [
        float(np.linalg.norm(vertebra_px[i + 1] - vertebra_px[i]))
        for i in range(len(VERTEBRAE) - 1)
    ]
    if not distances or np.mean(distances) < 1e-6:
        return pil_image

    mean_distance = float(np.mean(distances))
    draw = ImageDraw.Draw(pil_image, "RGBA")

    # 1. Thin vertebra-body outlines.
    for rect in _anatomy_body_rects(vertebra_px, disc_px, mean_distance):
        draw.rectangle(rect, outline=BODY_OUTLINE_COLOUR + (255,),
                       width=ANATOMY_LINE_WIDTH)

    # 2. Thin spinal-column axis connecting consecutive vertebra centres.
    for i in range(len(VERTEBRAE) - 1):
        p0 = (int(vertebra_px[i][0]), int(vertebra_px[i][1]))
        p1 = (int(vertebra_px[i + 1][0]), int(vertebra_px[i + 1][1]))
        draw.line([p0, p1], fill=AXIS_LINE_COLOUR + (200,),
                  width=ANATOMY_LINE_WIDTH)

    # 3. Thin horizontal offset lines (the exact displacement used for
    #    listhesis geometry) with the measured offset ratio label.
    font = _load_font(14)
    for i in range(len(VERTEBRAE) - 1):
        y = (float(vertebra_px[i][1]) + float(vertebra_px[i + 1][1])) / 2.0
        x0, x1 = (float(vertebra_px[i][0]), float(vertebra_px[i + 1][0]))
        offset = abs(x0 - x1) / mean_distance
        draw.line([(int(x0), int(y)), (int(x1), int(y))],
                  fill=OFFSET_LINE_COLOUR + (255,), width=ANATOMY_LINE_WIDTH)
        if offset_labels:
            draw.text((int((x0 + x1) / 2) - 6, int(y) - 16),
                      f"off {offset:.2f}", fill=OFFSET_LINE_COLOUR + (255,),
                      font=font, stroke_width=1, stroke_fill=(0, 0, 0, 200))

    return pil_image


def _position_label_boxes(placements, width: int, height: int):
    """
    Group label boxes by side, sort top-to-bottom and push overlapping
    neighbours apart so stacked labels never collide on the same side.
    """
    for side in ("right", "left"):
        group = sorted((p for p in placements if p["side"] == side),
                       key=lambda p: p["y0"])
        for a, b in zip(group, group[1:]):
            gap = a["y1"] + 4 - b["y0"]
            if gap > 0:
                b["y0"] += gap
                b["y1"] += gap


def draw_landmarks(
    image,
    coords,
    confidence=None,
    ground_truth=None,
    anatomy_lines: bool = DRAW_ANATOMY_THIN_LINES,
) -> Image.Image:
    """
    Draw all 10 predicted landmarks on the ORIGINAL image with a clean,
    non-overlapping overlay:

    * vertebrae as white-ringed circles, discs as white-ringed diamonds,
    * per-point labels (e.g. "L1  0.98") in dark rounded boxes on thin
      leader lines, alternating left/right with collision avoidance,
    * ground-truth as thin dashed-looking rings when provided,
    * a compact legend.

    When `anatomy_lines` is True (default) a thin-line vertebra-body
    outline + column-axis + offset-line schematic (derived purely from
    the detected keypoints) is drawn underneath first.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size

    pred_pixels = norm_to_pixels(coords, width, height).astype(np.float32)

    gt_pixels = None
    if ground_truth is not None:
        gt_pixels = norm_to_pixels(ground_truth, width, height).astype(np.float32)

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)

    num_vertebrae = len(VERTEBRAE)

    # Thin-line anatomical schematic first (under the markers).
    if anatomy_lines:
        try:
            pil_image = draw_thin_anatomy_overlay(
                pil_image, coords, offset_labels=False
            )
        except Exception:  # noqa: BLE001 - overlay must never crash the view
            pass

    draw = ImageDraw.Draw(pil_image, "RGBA")
    marker_r = 6
    label_font = _load_font(15)

    # Ground truth rings (underneath predictions).
    if gt_pixels is not None:
        for px, py in gt_pixels:
            draw.ellipse([px - 9, py - 9, px + 9, py + 9],
                         outline=(0, 0, 0, 255), width=4)
            draw.ellipse([px - 9, py - 9, px + 9, py + 9],
                         outline=GROUND_TRUTH_COLOUR + (255,), width=2)

    # Markers + label box placement.
    placements = []
    for i, (px, py) in enumerate(pred_pixels):
        is_vertebra = i < num_vertebrae
        colour = (
            VERTEBRA_COLOURS[i % len(VERTEBRA_COLOURS)]
            if is_vertebra
            else DISC_COLOURS[(i - num_vertebrae) % len(DISC_COLOURS)]
        )

        text = ALL_POINTS[i]
        if confidence is not None and i < len(confidence):
            text += f"  {confidence[i]:.2f}"

        # White-ringed coloured markers (read on dark and bright MRI alike).
        if is_vertebra:
            draw.ellipse([px - marker_r, py - marker_r,
                          px + marker_r, py + marker_r],
                         fill=(255, 255, 255, 255), outline=(0, 0, 0, 255),
                         width=1)
            draw.ellipse([px - marker_r + 2, py - marker_r + 2,
                          px + marker_r - 2, py + marker_r - 2],
                         fill=colour + (255,))
        else:
            r = marker_r + 1
            draw.polygon([(px, py - r), (px + r, py), (px, py + r),
                          (px - r, py)], fill=(255, 255, 255, 255))
            draw.polygon([(px, py - r + 2), (px + r - 2, py),
                          (px, py + r - 2), (px - r + 2, py)],
                         fill=colour + (255,))

        # Label side: alternate, but flip when the point hugs an image edge.
        side = "right" if i % 2 == 0 else "left"
        if px - 30 < 0:
            side = "right"
        elif px + 30 > width:
            side = "left"

        bbox = draw.textbbox((0, 0), text, font=label_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_h = th + 10
        box_w = tw + 18
        pad = 14

        cy = int(round(float(py)))
        if side == "right":
            x1 = min(int(round(px)) + marker_r + pad + box_w, width - 8)
            x0 = x1 - box_w
        else:
            x0 = max(int(round(px)) - marker_r - pad - box_w, 8)
            x1 = x0 + box_w

        placements.append({
            "px": int(round(float(px))), "py": cy, "side": side,
            "x0": x0, "x1": x1,
            "y0": cy - box_h // 2, "y1": cy + box_h // 2,
            "colour": colour, "text": text,
        })

    _position_label_boxes(placements, width, height)

    for p in placements:
        px, py = p["px"], p["py"]
        leader_end = (p["x0"] if p["side"] == "right" else p["x1"],
                      (p["y0"] + p["y1"]) // 2)
        leader_start = (px + marker_r if p["side"] == "right"
                        else px - marker_r, py)
        draw.line([leader_start, leader_end],
                  fill=p["colour"] + (160,), width=1)
        draw.rounded_rectangle([p["x0"], p["y0"], p["x1"], p["y1"]],
                               radius=6, fill=(8, 8, 8, 200),
                               outline=p["colour"] + (255,), width=1)
        draw.text((p["x0"] + 8, p["y0"] + 5), p["text"],
                  fill=(255, 255, 255, 255), font=label_font)

    # Legend
    legend_font = _load_font(14)
    x, y = 12, 12
    entries = [("Vertebra", VERTEBRA_COLOURS[0], "circle"),
               ("Disc", DISC_COLOURS[0], "diamond"),
               ("Vertebra outline (thin lines)", BODY_OUTLINE_COLOUR, "line")]
    if gt_pixels is not None:
        entries.append(("Ground truth", GROUND_TRUTH_COLOUR, "ring"))

    box_w = max(draw.textbbox((0, 0), t, font=legend_font)[2]
                for t, _, _ in entries) + 46
    box_h = 24 * len(entries) + 12
    draw.rounded_rectangle([x - 6, y - 6, x + box_w, y + box_h],
                           radius=8, fill=(8, 8, 8, 190))

    for text, colour, marker in entries:
        if marker == "diamond":
            draw.polygon([(x + 7, y - 3), (x + 13, y + 3), (x + 7, y + 9),
                          (x + 1, y + 3)], fill=(255, 255, 255, 255))
            draw.polygon([(x + 8, y - 2), (x + 12, y + 3), (x + 8, y + 8),
                          (x + 4, y + 3)], fill=colour + (255,))
        elif marker == "ring":
            draw.ellipse([x + 1, y - 4, x + 15, y + 10],
                         outline=(255, 255, 255, 255), width=2)
            draw.ellipse([x + 2, y - 3, x + 14, y + 9],
                         outline=colour + (255,), width=1)
        elif marker == "line":
            draw.line([(x + 1, y + 2), (x + 15, y + 2)],
                      fill=colour + (255,), width=1)
        else:
            draw.ellipse([x + 1, y - 4, x + 15, y + 10],
                         fill=(255, 255, 255, 255))
            draw.ellipse([x + 3, y - 2, x + 13, y + 8], fill=colour + (255,))
        draw.text((x + 22, y - 1), text, fill=(255, 255, 255, 255),
                  font=legend_font)
        y += 24

    return pil_image


MEDICAL_DISCLAIMER = "Research prototype — not a medical diagnostic tool."
