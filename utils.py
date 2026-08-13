"""
utils.py
========

Shared helpers for the Spine Foundation Model:

- Preprocessing (must match training: Resize 512 -> ToTensor)
- Model / checkpoint loading
- Single-image inference
- Landmark visualization on the ORIGINAL image dimensions
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from config import BEST_MODEL, IMAGE_SIZE, RELEASE_MODEL
from model import SpineFoundationModel

# Lumbar levels in fixed order (matches dataset/model head order)
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

# One marker colour per level (bright colours that stand out on MRI)
LEVEL_COLOURS = [
    (0, 229, 255),    # cyan
    (80, 255, 120),   # green
    (255, 200, 0),    # amber
    (255, 100, 255),  # magenta
    (255, 130, 70),   # orange
]

PREDICTION_COLOUR = (255, 60, 60)      # red marker for prediction
GROUND_TRUTH_COLOUR = (60, 255, 60)    # green marker for ground truth


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
    """Convert a PIL image into the (3, 512, 512) tensor the model expects."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return get_transform()(image)


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------

def load_model(
    checkpoint_path: Union[str, Path] = BEST_MODEL,
    device: Optional[torch.device] = None,
):
    """
    Load the Spine Foundation Model from a training checkpoint.

    Falls back to the committed release model (fp16, model-only) when the
    requested checkpoint does not exist, so a fresh clone works out of the box.

    Returns
    -------
    (model, device, checkpoint_path)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists() and RELEASE_MODEL.exists():
        checkpoint_path = RELEASE_MODEL

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained model checkpoint found: {checkpoint_path}"
        )

    model = SpineFoundationModel().to(device)
    model.eval()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict", checkpoint
    )

    first_tensor = next(iter(state_dict.values()))

    if first_tensor.dtype == torch.float16:
        model = model.half()

    model.load_state_dict(state_dict)

    return model, device, checkpoint_path


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

@torch.no_grad()
def predict(
    model: SpineFoundationModel,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run inference on a single preprocessed image tensor.

    Returns a dict with:
        coords      : Tensor(1, 10) normalized x,y per level
        confidence  : Tensor(1, 5)
        features    : Tensor(1, 512)
    """
    input_batch = image_tensor.unsqueeze(0).to(device)

    if next(model.parameters()).dtype == torch.float16:
        input_batch = input_batch.half()

    return model(input_batch)


def norm_to_pixels(
    coords: Union[torch.Tensor, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert normalized (0-1) coordinates to pixel coordinates
    using the ORIGINAL image dimensions.
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    pixels = np.empty_like(coords)
    pixels[:, 0] = coords[:, 0] * width
    pixels[:, 1] = coords[:, 1] * height
    return pixels


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

def _to_pil_image(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Coerce a PIL image or numpy array into a PIL RGB image."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image.convert("RGB")


def _draw_legend(draw: ImageDraw.ImageDraw, has_ground_truth: bool) -> None:
    """Draw a small legend explaining marker colours."""
    x, y = 12, 12

    box = draw.textbbox((x, y), "Prediction", font=_load_font(18))
    draw.rectangle(
        [x - 6, y - 6, x + box[2] + 22, y + box[3] + 6],
        fill=(0, 0, 0),
    )
    draw.ellipse([x + 2, y + 2, x + 14, y + 14], fill=PREDICTION_COLOUR)
    draw.text((x + 20, y), "Prediction", fill=(255, 255, 255))

    if has_ground_truth:
        y += 26
        box = draw.textbbox((x, y), "Ground Truth", font=_load_font(18))
        draw.rectangle(
            [x - 6, y - 6, x + box[2] + 22, y + box[3] + 6],
            fill=(0, 0, 0),
        )
        draw.ellipse(
            [x + 2, y + 2, x + 14, y + 14],
            outline=GROUND_TRUTH_COLOUR,
            width=3,
        )
        draw.text((x + 20, y), "Ground Truth", fill=(255, 255, 255))


def _load_font(size: int):
    """Return a default font, trying a scalable size when supported."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow versions
        return ImageFont.load_default()


def draw_landmarks(
    image: Union[Image.Image, np.ndarray],
    coords: Union[torch.Tensor, np.ndarray, list],
    confidence: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
    ground_truth: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
) -> Image.Image:
    """
    Draw the five lumbar landmark predictions on the ORIGINAL image.

    Parameters
    ----------
    image       : PIL image or numpy array (original dimensions).
    coords      : normalized (0-1) coordinates, shape (10,) -> [x,y] per level.
    confidence  : optional shape (5,) confidence values shown with labels.
    ground_truth: optional normalized (0-1) coordinates for comparison.

    Returns
    -------
    A new PIL image with landmarks drawn. The input image is not modified.
    """
    pil_image = _to_pil_image(image)

    width, height = pil_image.size

    pred_pixels = norm_to_pixels(coords, width, height)

    gt_pixels = None
    if ground_truth is not None:
        gt_pixels = norm_to_pixels(ground_truth, width, height)

    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)

    draw = ImageDraw.Draw(pil_image, "RGBA")

    # Ground truth markers first (so predictions stay on top)
    if gt_pixels is not None:
        for i, (px, py) in enumerate(gt_pixels):
            r = 9
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                outline=GROUND_TRUTH_COLOUR + (255,),
                width=3,
            )

    # Predicted markers + labels
    for i, (px, py) in enumerate(pred_pixels):

        colour = LEVEL_COLOURS[i % len(LEVEL_COLOURS)]
        r = 10

        # filled marker with contrasting outline
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=colour + (255,),
            outline=(0, 0, 0),
            width=2,
        )

        label = LEVELS[i]
        if confidence is not None and i < len(confidence):
            label += f" {confidence[i]:.2f}"

        font = _load_font(22)
        text_x = px + r + 6
        text_y = py - r - 4

        # clip label within the image bounds
        text_x = min(text_x, width - 10)
        text_y = min(text_y, height - 10)

        draw.text(
            (text_x, text_y),
            label,
            fill=colour + (255,),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    _draw_legend(draw, has_ground_truth=gt_pixels is not None)

    return pil_image


# ---------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------

MEDICAL_DISCLAIMER = "Research prototype — not a medical diagnostic tool."
