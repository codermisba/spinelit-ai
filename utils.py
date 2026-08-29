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


def draw_landmarks(
    image,
    coords,
    confidence=None,
    ground_truth=None,
) -> Image.Image:
    """
    Draw all 10 landmark predictions (vertebrae as circles, discs as
    diamonds) on the ORIGINAL image with per-point confidence labels.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size

    pred_pixels = norm_to_pixels(coords, width, height)

    gt_pixels = None
    if ground_truth is not None:
        gt_pixels = norm_to_pixels(ground_truth, width, height)

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)

    num_vertebrae = len(VERTEBRAE)

    draw = ImageDraw.Draw(pil_image, "RGBA")

    font = _load_font(20)

    def _label_point(px, py, text, colour, marker="circle"):
        if marker == "diamond":
            r = 9
            draw.polygon(
                [(px, py - r), (px + r, py), (px, py + r), (px - r, py)],
                fill=colour + (255,),
                outline=(0, 0, 0),
            )
        else:
            r = 10
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=colour + (255,),
                outline=(0, 0, 0),
                width=2,
            )

        text_x = min(px + r + 5, width - 10)
        text_y = min(py - r - 4, max(4, height - 30))
        draw.text(
            (text_x, text_y), text,
            fill=colour + (255,), font=font,
            stroke_width=2, stroke_fill=(0, 0, 0),
        )

    # Ground truth first (predictions stay on top)
    if gt_pixels is not None:
        for px, py in gt_pixels:
            draw.ellipse(
                [px - 9, py - 9, px + 9, py + 9],
                outline=GROUND_TRUTH_COLOUR + (255,), width=3,
            )

    for i, (px, py) in enumerate(pred_pixels):

        is_vertebra = i < num_vertebrae

        colour = (
            VERTEBRA_COLOURS[i % len(VERTEBRA_COLOURS)]
            if is_vertebra
            else DISC_COLOURS[(i - num_vertebrae) % len(DISC_COLOURS)]
        )

        label = ALL_POINTS[i]
        if confidence is not None and i < len(confidence):
            label += f" {confidence[i]:.2f}"

        _label_point(
            px, py, label, colour,
            marker="circle" if is_vertebra else "diamond",
        )

    # Legend
    legend_font = _load_font(16)
    x, y = 12, 12
    entries = [("Vertebra centre", VERTEBRA_COLOURS[0], "circle"),
               ("Disc centre", DISC_COLOURS[0], "diamond")]
    if gt_pixels is not None:
        entries.append(("Ground truth", GROUND_TRUTH_COLOUR, "ring"))

    box_w = max(draw.textbbox((0, 0), t, font=legend_font)[2]
                for t, _, _ in entries) + 46
    box_h = 26 * len(entries) + 12
    draw.rectangle([x - 6, y - 6, x + box_w, y + box_h], fill=(0, 0, 0, 200))

    for text, colour, marker in entries:
        if marker == "diamond":
            draw.polygon([(x + 7, y - 3), (x + 13, y + 3), (x + 7, y + 9),
                          (x + 1, y + 3)], fill=colour + (255,))
        elif marker == "ring":
            draw.ellipse([x + 1, y - 3, x + 13, y + 9],
                         outline=colour + (255,), width=3)
        else:
            draw.ellipse([x + 1, y - 3, x + 13, y + 9], fill=colour + (255,))
        draw.text((x + 22, y - 2), text, fill=(255, 255, 255),
                  font=legend_font)
        y += 26

    return pil_image


MEDICAL_DISCLAIMER = "Research prototype — not a medical diagnostic tool."
