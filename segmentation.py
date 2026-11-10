"""
segmentation.py
===============

Standalone 2-D U-Net segmentation for sagittal lumbar MRI.

Why a separate model
--------------------
The rebuilt ``SpineFoundationModel`` also has an optional 4-class
segmentation head (trained jointly when masks exist), but the reliable path
for **real** masks is a dedicated segmentation network trained on
nnU-Net-style data (see ``train_segmentation.py``).  When a segmentation
checkpoint is present the API uses it to draw true vertebra / disc /
spinal-canal masks; when it is absent the API falls back to
landmark-derived blocks and says so.

Classes (must match ``config.SEG_CLASS_NAMES``)
-----------------------------------------------
    0 background
    1 vertebra
    2 disc
    3 spinal canal

Usage
-----
    from segmentation import SegmentationEngine
    seg = SegmentationEngine()               # loads checkpoints/segmentation.pth
    if seg.available:
        result = seg.analyze(image)
        result["mask"]        # HxW uint8 at ORIGINAL resolution
        result["probabilities"]# CxHxW float32
        result["boxes"]       # per-class pixel boxes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import config as _cfg
from config import IMAGENET_MEAN, IMAGENET_STD, NUM_SEG_CLASSES, SEG_CHECKPOINT


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


class UNet(nn.Module):
    """Compact U-Net (GroupNorm so batch size 1-2 is safe on CPU)."""

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_SEG_CLASSES,
                 base_channels: int = 32):
        super().__init__()
        c = int(base_channels)
        self.enc1 = _conv_block(in_channels, c)
        self.enc2 = _conv_block(c, c * 2)
        self.enc3 = _conv_block(c * 2, c * 4)
        self.enc4 = _conv_block(c * 4, c * 8)
        self.bottleneck = _conv_block(c * 8, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c * 8, c * 8, 2, stride=2)
        self.dec4 = _conv_block(c * 16, c * 4)
        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = _conv_block(c * 8, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = _conv_block(c * 4, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = _conv_block(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ----------------------------------------------------------------------
# Pre/post processing
# ----------------------------------------------------------------------

def segmentation_transform(image_size: int | None = None):
    """RGB resize + ImageNet normalisation (same convention as the localizer)."""
    from torchvision import transforms

    size = int(image_size or _cfg.IMAGE_SIZE)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _hub_segmentation_path() -> Optional[Path]:
    """Download segmentation.pth from MODEL_ID when no local file exists.

    Hosted deployments exclude ``checkpoints/*.pth`` from the image, so the
    standalone U-Net has to come from the same Hub repo as the vision model.
    Only attempted when the deployment is configured with a Hub repo id;
    failures are non-fatal because segmentation is an optional component.
    """
    try:
        from weights import download_sidecar
        filename = os.environ.get("SEG_MODEL_FILENAME", "segmentation.pth")
        return download_sidecar(filename)
    except Exception:  # noqa: BLE001 - optional task, never fatal
        return None


def load_segmentation_model(
    checkpoint_path: str | Path = SEG_CHECKPOINT,
    device: Optional[torch.device] = None,
):
    """Load a segmentation checkpoint saved by ``train_segmentation.py``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        hub_path = _hub_segmentation_path()
        if hub_path is not None:
            checkpoint_path = hub_path
        raise FileNotFoundError(
            f"No segmentation checkpoint at {checkpoint_path}. Train one "
            "with: python train_segmentation.py --images ... --masks ..."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    num_classes = int(meta.get("num_classes", NUM_SEG_CLASSES))
    base_channels = int(meta.get("base_channels", 32))
    image_size = int(meta.get("image_size", _cfg.IMAGE_SIZE))

    model = UNet(num_classes=num_classes, base_channels=base_channels).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, device, checkpoint_path, {
        "num_classes": num_classes,
        "base_channels": base_channels,
        "image_size": image_size,
        "version": meta.get("version", ""),
        "created_at": meta.get("created_at", ""),
        "val_dice": meta.get("val_dice"),
    }


@torch.no_grad()
def segment_image(model, image: Image.Image, device: torch.device,
                  image_size: int | None = None) -> dict:
    """
    Run segmentation and return probabilistic + hard masks at ORIGINAL size.

    Returns
    -------
    dict with
        probabilities : (C,H,W) float32 in [0,1]
        mask          : (H,W) uint8 class indices
        overlay       : (H,W,3) uint8 colour image (background transparent)
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    tensor = segmentation_transform(image_size)(rgb).unsqueeze(0).to(device)

    model_dtype = next(model.parameters()).dtype
    if tensor.dtype != model_dtype:
        tensor = tensor.to(model_dtype)

    logits = model(tensor)
    probs = torch.softmax(logits.float(), dim=1)
    probs = F.interpolate(probs, size=(height, width), mode="bilinear",
                          align_corners=False)
    probs_np = probs[0].cpu().numpy()
    mask = probs_np.argmax(axis=0).astype(np.uint8)

    colours = _class_colour_lut()
    overlay = colours[mask]
    return {
        "probabilities": probs_np,
        "mask": mask,
        "overlay": overlay,
    }


def _class_colour_lut() -> np.ndarray:
    names = list(_cfg.SEG_CLASS_NAMES)
    configured = dict(_cfg.SEG_CLASS_COLOURS)
    lut = np.zeros((max(len(names), 4), 3), dtype=np.uint8)
    lut[0] = (0, 0, 0)
    if "vertebra" in names:
        lut[names.index("vertebra")] = configured.get(
            "vertebra", (80, 200, 255))
    if "disc" in names:
        lut[names.index("disc")] = configured.get("disc", (255, 200, 60))
    if "spinal_canal" in names:
        lut[names.index("spinal_canal")] = configured.get(
            "spinal_canal", (140, 255, 140))
    return lut


def mask_boxes(mask: np.ndarray, min_area: int = 40) -> list[dict]:
    """
    Connected-component boxes per foreground class (pure numpy, no scipy).

    Returns ``[{class_id, class_name, box:[x0,y0,x1,y1], area}]``.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        return []
    results: list[dict] = []
    names = list(_cfg.SEG_CLASS_NAMES)
    for class_id in range(1, int(mask.max()) + 1):
        binary = mask == class_id
        if not binary.any():
            continue
        for box, area in _connected_boxes(binary):
            if area < int(min_area):
                continue
            results.append({
                "class_id": class_id,
                "class_name": names[class_id] if class_id < len(names)
                              else f"class_{class_id}",
                "box": [float(v) for v in box],
                "area": int(area),
                "source": "segmentation",
            })
    return results


def _connected_boxes(binary: np.ndarray):
    """4-connected components via an iterative flood fill."""
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] \
                            and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            yield [x0, y0, x1, y1], area


class SegmentationEngine:
    """Lazy wrapper used by the vision engine / API (optional component)."""

    def __init__(self, checkpoint: Optional[str | Path] = None):
        self._model = None
        self._device = None
        self._path = None
        self._meta: dict = {}
        self.error = ""
        self.load(checkpoint)

    def load(self, checkpoint: Optional[str | Path] = None):
        path = Path(checkpoint or SEG_CHECKPOINT)
        try:
            self._model, self._device, self._path, self._meta = \
                load_segmentation_model(path)
            self.error = ""
        except Exception as exc:  # noqa: BLE001 - optional component
            self._model = None
            self._device = None
            self._path = None
            self._meta = {}
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    def status(self) -> dict:
        return {
            "loaded": self.available,
            "checkpoint": str(self._path) if self._path else "none",
            "num_classes": self._meta.get("num_classes"),
            "version": self._meta.get("version", ""),
            "val_dice": self._meta.get("val_dice"),
            "error": self.error,
        }

    def analyze(self, image: Image.Image) -> dict:
        if not self.available:
            raise RuntimeError(self.error or "Segmentation model not loaded.")
        out = segment_image(
            self._model, image, self._device,
            image_size=self._meta.get("image_size"),
        )
        out["boxes"] = mask_boxes(out["mask"])
        return out
