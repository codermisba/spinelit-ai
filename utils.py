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

import config as _cfg
from config import (
    BEST_MODEL,
    DDD_LABELS_CSV,
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
    """Return the exact transform used during training.

    `IMAGE_SIZE` is read through the config module (not a bound import) so
    a loaded checkpoint can override it with its own training-time size.
    ImageNet normalisation must match training exactly — the pretrained
    ConvNeXt backbone was trained with it, and inference without it silently
    shifts every feature.
    """
    size = _cfg.IMAGE_SIZE
    steps = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
    if getattr(_cfg, "USE_IMAGENET_NORM", True):
        steps.append(
            transforms.Normalize(_cfg.IMAGENET_MEAN, _cfg.IMAGENET_STD)
        )
    return transforms.Compose(steps)


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

    Every checkpoint saved after ``train.py`` carries a ``meta`` block with
    the exact training hyper-parameters that affect the network shape and
    the input preprocessing (``image_size``, ``disc_feature_stage``,
    ``model_name``, ...).  These override the `config` module for the rest
    of the process, so the same binary can serve checkpoints that were
    trained at different image sizes / feature stages WITHOUT an env change
    a restart.  Old checkpoints without ``meta`` fall back to config.py.

    Returns
    -------
    (model, device, checkpoint_path, meta)
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

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)

    raw_meta = checkpoint.get("meta") or {}
    meta = _apply_checkpoint_meta(raw_meta)

    # Task flags tell inference which heads are actually trustworthy.  Legacy
    # checkpoints without `tasks` are treated conservatively: only the
    # localizer is assumed trained.
    tasks = checkpoint.get("tasks") or raw_meta.get("tasks") or {}
    meta["tasks"] = {
        "localization": True,
        "ddd": bool(tasks.get("ddd", False)),
        "seg": bool(tasks.get("seg", False)),
    }

    # Weights come from the checkpoint; no need to fetch ImageNet weights.
    _cfg.PRETRAINED = False

    model = SpineFoundationModel().to(device)
    model.eval()

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Unrecognised checkpoint layout in {checkpoint_path}."
        )

    # Cast half-precision artifacts to fp32 on CPU.  fp16 convolution on CPU
    # is either unsupported or dramatically slower, and silently producing
    # NaN features is worse than the small memory cost of fp32.
    contains_half = any(
        torch.is_tensor(v) and v.dtype == torch.float16
        for v in state_dict.values()
    )
    if contains_half:
        if device.type == "cuda":
            model = model.half()
        else:
            state_dict = {
                k: (v.float() if torch.is_tensor(v)
                    and v.dtype == torch.float16 else v)
                for k, v in state_dict.items()
            }

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # The DDD head is retrained independently (finetune_ddd.py) and may
        # legitimately gain parameters (e.g. the raw-image patch encoder).
        # Allow ONLY disc_head differences; every other tensor must match
        # exactly, so a genuinely incompatible checkpoint still fails loudly.
        current = model.state_dict()
        head_prefix = "disc_head."
        incompatible = []
        for key in set(current) | set(state_dict):
            if key in current and key in state_dict:
                if current[key].shape != state_dict[key].shape:
                    incompatible.append(key)
            else:
                incompatible.append(key)
        if any(not key.startswith(head_prefix) for key in incompatible):
            raise
        filtered = {
            key: value for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        dropped = [key for key in incompatible if key.startswith(head_prefix)]
        if dropped and meta["tasks"]["ddd"]:
            # The checkpoint claims a trained head but its tensors do not fit
            # this architecture.  Loading would leave parts of the head
            # randomly initialised, which would surface as fabricated grades.
            raise RuntimeError(
                f"{checkpoint_path} declares a trained DDD head but its "
                f"disc_head weights do not match the current architecture "
                f"({len(dropped)} mismatched tensors). Retrain it with "
                f"finetune_ddd.py instead of loading stale weights."
            ) from None
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        remaining = [key for key in list(missing) + list(unexpected)
                     if not key.startswith(head_prefix)]
        if remaining:
            raise
    model.eval()

    return model, device, checkpoint_path, meta


def _apply_checkpoint_meta(meta: dict) -> dict:
    """
    Apply per-checkpoint settings (image size, backbone, head sizes,
    normalisation) to the config module so model construction and
    preprocessing match the checkpoint.

    Returns the sanitized meta dict (all scalar keys coerced to primitives)
    for the caller to expose as the model version / status.  Structured
    values such as `tasks` are intentionally not copied into the scalar
    status dict; callers that need them read the raw checkpoint.
    """
    selected = (
        "image_size",
        "disc_feature_stage",
        "num_keypoints",
        "num_discs",
        "num_pfrrmann_classes",
        "num_seg_classes",
        "fpn_dim",
        "heatmap_stride",
        "softargmax_temperature",
        "disc_pool_sigma_fraction",
        "use_imagenet_norm",
        "model_name",
    )
    out: dict = {}
    for key in selected:
        if key in meta and meta[key] is not None:
            out[key] = meta[key]

    if "image_size" in out:
        _cfg.IMAGE_SIZE = int(out["image_size"])
    if "disc_feature_stage" in out:
        _cfg.DISC_FEATURE_STAGE = int(out["disc_feature_stage"])
    if "model_name" in out:
        _cfg.MODEL_NAME = str(out["model_name"])
    if "fpn_dim" in out:
        _cfg.FPN_DIM = int(out["fpn_dim"])
    if "softargmax_temperature" in out:
        _cfg.SOFTARGMAX_TEMPERATURE = float(out["softargmax_temperature"])
    if "disc_pool_sigma_fraction" in out:
        _cfg.DISC_POOL_SIGMA_FRACTION = float(
            out["disc_pool_sigma_fraction"])
    if "num_seg_classes" in out:
        _cfg.NUM_SEG_CLASSES = int(out["num_seg_classes"])
    if "use_imagenet_norm" in out:
        _cfg.USE_IMAGENET_NORM = bool(out["use_imagenet_norm"])

    scalar = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalar[key] = value
    out["created_at"] = scalar.get("created_at", "")
    out["version"] = scalar.get("version", "")
    return out


# ---------------------------------------------------------
# Inference + decoding
# ---------------------------------------------------------

@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device,
            disc_centers=None, return_seg: bool = True) -> dict:
    """Run inference on a single preprocessed image tensor."""
    input_batch = image_tensor.unsqueeze(0).to(device)

    if next(model.parameters()).dtype == torch.float16:
        input_batch = input_batch.half()

    if disc_centers is not None:
        disc_centers = torch.as_tensor(
            disc_centers, dtype=torch.float32, device=device
        ).reshape(1, -1, 2)
        if next(model.parameters()).dtype == torch.float16:
            disc_centers = disc_centers.half()

    return model(input_batch, disc_centers=disc_centers,
                 return_seg=return_seg)


def decode_outputs(outputs: dict) -> dict:
    """
    Convert raw model outputs into interpretable numpy values:

        points             : (10,2) normalized coordinates
        localization_conf  : (10,) 0-1
        ddd_prob           : (5,5) softmax Pfirrmann class probabilities
        ddd_grade          : (5,) predicted Pfirrmann grade (1-5)
        ddd_conf           : (5,) 0-1 maximum class probability per disc
        heatmap_peak       : (10,) peak heatmap probability (localization)
        seg_probs          : (C,Hf,Wf) or None
    """
    import numpy as np

    logits = outputs["ddd_logits"].float().detach().cpu().numpy()
    # Squeeze any leading batch dim: predict() returns (1,5,5).
    logits = logits.reshape(-1, logits.shape[-1])          # (5, 5) per disc
    probs = _softmax(logits, axis=-1)

    decoded = {
        "points": outputs["coords"].float().detach().cpu().numpy()
                  .reshape(NUM_KEYPOINTS, 2),
        "localization_conf": outputs["localization_conf"].float().detach().cpu()
                             .numpy().reshape(NUM_KEYPOINTS),
        "ddd_prob": probs,
        "ddd_grade": (np.argmax(probs, axis=-1) + 1.0).reshape(-1),
        "ddd_conf": probs.max(axis=-1).reshape(-1),
    }

    if "heatmap_peak" in outputs:
        decoded["heatmap_peak"] = (
            outputs["heatmap_peak"].float().detach().cpu().numpy()
            .reshape(NUM_KEYPOINTS)
        )

    if outputs.get("seg_logits") is not None:
        seg = outputs["seg_logits"].float().detach()
        seg_probs = torch.softmax(seg, dim=1)[0].cpu().numpy()
        decoded["seg_probs"] = seg_probs
        decoded["seg_mask"] = seg_probs.argmax(axis=0).astype(np.uint8)

    return decoded


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


def _position_label_boxes(placements, width: int, height: int,
                          markers=None, marker_r: int = 8):
    """
    Group label boxes by side, sort top-to-bottom and push overlapping
    neighbours apart so stacked labels never collide on the same side.

    ``markers`` (optional) is a list of ``(x, y)`` landmark pixel centres.
    Those are treated as fixed obstacles too, so a pushed-down label box
    cannot sit on top of a landmark marker (a purely cosmetic problem that
    made some markers unreadable in dense overlays).
    """
    obstacles = [
        (x - marker_r, y - marker_r, x + marker_r, y + marker_r)
        for x, y in (markers or [])
    ]

    def collides(box, rects) -> bool:
        x0, y0, x1, y1 = box
        return any(x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1
                   for a0, b0, a1, b1 in rects)

    for side in ("right", "left"):
        group = sorted((p for p in placements if p["side"] == side),
                       key=lambda p: p["y0"])
        placed = []
        for p in group:
            box_h = p["y1"] - p["y0"]
            start = max(4, min(height - 4 - box_h, p["y0"]))
            p["y0"], p["y1"] = start, start + box_h

            # Prefer moving down; if that would leave the canvas, move up.
            for _ in range(120):
                if not collides((p["x0"], p["y0"], p["x1"], p["y1"]),
                                obstacles + placed):
                    break
                p["y0"] += 3
                p["y1"] += 3
            if p["y1"] > height - 4:
                p["y0"] = max(4, min(start, height - 4 - box_h))
                p["y1"] = p["y0"] + box_h
                for _ in range(120):
                    if not collides((p["x0"], p["y0"], p["x1"], p["y1"]),
                                    obstacles + placed):
                        break
                    p["y0"] = max(4, p["y0"] - 3)
                    p["y1"] = p["y0"] + box_h
            placed.append((p["x0"], p["y0"], p["x1"], p["y1"]))


def _draw_landmark_legend(draw, gt_present: bool) -> None:
    """Draw the compact landmark legend (background layer).

    Called *before* the markers/labels so it can never hide a landmark in
    the top-left corner of the image.
    """
    legend_font = _load_font(14)
    x, y = 12, 12
    entries = [("Vertebra", VERTEBRA_COLOURS[0], "circle"),
               ("Disc", DISC_COLOURS[0], "diamond"),
               ("Vertebra outline (thin lines)", BODY_OUTLINE_COLOUR, "line")]
    if gt_present:
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

    # Legend goes down first so markers/labels always remain readable.
    _draw_landmark_legend(draw, gt_present=gt_pixels is not None)

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

    marker_centres = [
        (int(round(float(px))), int(round(float(py))))
        for px, py in pred_pixels
    ]
    _position_label_boxes(placements, width, height, markers=marker_centres)

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

    return pil_image


# ---------------------------------------------------------
# Block / detection / segmentation rendering
# ---------------------------------------------------------

def landmark_boxes(coords, width: int, height: int,
                   include_discs: bool = True) -> list[dict]:
    """
    Build axis-aligned blocks (in ORIGINAL pixel coordinates) purely from
    the 10 predicted landmarks.

    Vertebra blocks use the disc centres above/below each vertebral body
    (the same construction as the thin-line anatomy overlay).  Disc blocks
    are small boxes centred on each predicted disc centre.  These are
    *landmark-derived* blocks, not a learned detector/segmentation output.
    """
    points_px = norm_to_pixels(coords, width, height).astype(np.float64)
    vert_px = points_px[:len(VERTEBRAE)]
    disc_px = points_px[len(VERTEBRAE):]

    distances = [
        float(np.linalg.norm(vert_px[i + 1] - vert_px[i]))
        for i in range(len(VERTEBRAE) - 1)
    ]
    if not distances or np.mean(distances) < 1e-6:
        return []
    mean_distance = float(np.mean(distances))

    boxes = []
    for label, rect in zip(
        VERTEBRAE, _anatomy_body_rects(vert_px, disc_px, mean_distance)
    ):
        boxes.append({
            "label": label,
            "class_name": "vertebra",
            "source": "landmark",
            "box": [float(v) for v in rect],
        })

    if include_discs:
        for label, (cx, cy) in zip(DISC_LEVELS, disc_px):
            half_w = 0.46 * mean_distance
            half_h = 0.16 * mean_distance
            boxes.append({
                "label": label,
                "class_name": "disc",
                "source": "landmark",
                "box": [float(cx - half_w), float(cy - half_h),
                        float(cx + half_w), float(cy + half_h)],
            })
    return boxes


def _clamp_box(box, width: int, height: int):
    x0, y0, x1, y1 = [float(v) for v in box]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (max(0, min(width - 1, x0)), max(0, min(height - 1, y0)),
            max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))


def draw_blocks(
    image,
    coords=None,
    boxes: Optional[list[dict]] = None,
    confidence=None,
    fill_alpha: int = 34,
    show_labels: bool = True,
) -> Image.Image:
    """
    Draw the localizer's vertebra/disc *blocks* on the original image.

    If `boxes` is omitted, blocks are derived from the predicted keypoints
    with :func:`landmark_boxes`.  Each box may carry
    ``label``/``class_name``/``confidence``/``source`` keys.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size
    if boxes is None:
        if coords is None:
            return pil_image
        boxes = landmark_boxes(coords, width, height)
    if not boxes:
        return pil_image

    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _load_font(14)
    num_vert = len(VERTEBRAE)

    for i, item in enumerate(boxes):
        box = item.get("box")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, width, height)
        cls = str(item.get("class_name", "vertebra"))
        is_disc = cls == "disc"
        if "colour" in item:
            colour = tuple(item["colour"])[:3]
        elif is_disc:
            colour = DISC_COLOURS[(i - num_vert) % len(DISC_COLOURS)]
        else:
            colour = VERTEBRA_COLOURS[i % len(VERTEBRA_COLOURS)]

        draw.rectangle([x0, y0, x1, y1],
                       fill=colour + (fill_alpha,),
                       outline=colour + (255,), width=2)

        if show_labels:
            label = str(item.get("label", ""))
            conf = item.get("confidence")
            if conf is None and confidence is not None and not is_disc:
                if i < len(confidence):
                    conf = float(confidence[i])
            if conf is not None:
                label = f"{label} {float(conf):.2f}".strip()
            if label:
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ly = max(2, y0 - th - 6)
                draw.rounded_rectangle(
                    [x0, ly, x0 + tw + 10, ly + th + 6], radius=4,
                    fill=(8, 8, 8, 210),
                )
                draw.text((x0 + 5, ly + 2), label,
                          fill=(255, 255, 255, 255), font=font)
    return pil_image


def draw_detections(
    image,
    detections: list[dict],
    show_labels: bool = True,
) -> Image.Image:
    """
    Draw detector output on the original image.

    Each detection is ``{box:[x0,y0,x1,y1] (pixels), label?, class_name?,
    confidence?, source?}``.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size
    if not detections:
        return pil_image

    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = _load_font(14)
    for det in detections:
        box = det.get("box")
        if not box:
            continue
        x0, y0, x1, y1 = _clamp_box(box, width, height)
        cls = str(det.get("class_name", "vertebra"))
        is_disc = cls == "disc"
        colour = ((255, 200, 60) if is_disc else (110, 200, 255))
        draw.rectangle([x0, y0, x1, y1],
                       fill=colour + (26,), outline=colour + (255,), width=2)
        if show_labels:
            label = str(det.get("label", cls))
            conf = det.get("confidence")
            if conf is not None:
                label = f"{label} {float(conf):.2f}"
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ly = max(2, y0 - th - 6)
            draw.rounded_rectangle([x0, ly, x0 + tw + 10, ly + th + 6],
                                   radius=4, fill=(8, 8, 8, 210))
            draw.text((x0 + 5, ly + 2), label,
                      fill=(255, 255, 255, 255), font=font)
    return pil_image


def draw_segmentation_overlay(
    image,
    mask,
    class_names: Optional[list[str]] = None,
    class_colours: Optional[dict] = None,
    alpha: float = 0.38,
) -> Image.Image:
    """
    Alpha-blend a class mask (H,W ints) over the original image.

    `mask` is resized to the image with nearest-neighbour so class edges
    stay crisp.  Background (class 0) is left transparent.
    """
    pil_image = _to_pil_image(image)
    width, height = pil_image.size
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2 or mask_arr.size == 0:
        return pil_image

    mask_img = Image.fromarray(mask_arr.astype(np.uint8), mode="L").resize(
        (width, height), resample=Image.NEAREST
    )
    mask_arr = np.asarray(mask_img)

    names = class_names or list(getattr(_cfg, "SEG_CLASS_NAMES", []))
    colours = class_colours or dict(
        getattr(_cfg, "SEG_CLASS_COLOURS", {}) or {}
    )
    palette = {
        0: (0, 0, 0),
        1: tuple(colours.get("vertebra", (80, 200, 255))),
        2: tuple(colours.get("disc", (255, 200, 60))),
        3: tuple(colours.get("spinal_canal", (140, 255, 140))),
    }
    # Support masks with more than 4 classes by cycling a small palette.
    extra = [(255, 120, 120), (180, 120, 255), (120, 255, 220),
             (255, 160, 60), (200, 200, 120)]
    for idx, colour in enumerate(extra, start=len(palette)):
        palette[idx] = colour

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    visible = mask_arr > 0
    for idx, colour in palette.items():
        if idx == 0:
            continue
        sel = mask_arr == idx
        if sel.any():
            rgba[sel, :3] = colour
            rgba[sel, 3] = int(255 * alpha)

    overlay = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert(
        "RGB"
    )


def render_analysis(
    image,
    coords=None,
    confidence=None,
    boxes: Optional[list[dict]] = None,
    detections: Optional[list[dict]] = None,
    segmentation=None,
    gradcam=None,
    gradcam_alpha: float = 0.5,
    show_blocks: bool = True,
    show_landmarks: bool = True,
) -> Image.Image:
    """
    Compose the full analysis image used by the API/UI:

        1. optional Grad-CAM heatmap (bottom layer)
        2. optional segmentation mask overlay
        3. vertebra/disc blocks (keypoint-derived or detector output)
        4. keypoint markers + labels (top layer)

    Every layer is optional; the function degrades to whatever inputs exist.
    """
    pil_image = _to_pil_image(image)

    if gradcam is not None:
        try:
            from explainability import overlay_heatmap
            pil_image = overlay_heatmap(
                pil_image, np.asarray(gradcam), alpha=gradcam_alpha
            )
        except Exception:  # noqa: BLE001 - XAI must never break inference
            pass

    if segmentation is not None:
        try:
            pil_image = draw_segmentation_overlay(pil_image, segmentation)
        except Exception:  # noqa: BLE001
            pass

    if detections:
        pil_image = draw_detections(pil_image, detections)
    elif show_blocks and (boxes is not None or coords is not None):
        pil_image = draw_blocks(
            pil_image, coords=coords, boxes=boxes, confidence=confidence
        )

    if show_landmarks and coords is not None:
        pil_image = draw_landmarks(
            pil_image, coords=coords, confidence=confidence,
            anatomy_lines=False,
        )
    return pil_image


MEDICAL_DISCLAIMER = "Research prototype — not a medical diagnostic tool."
