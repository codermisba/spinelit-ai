"""
explainability.py
=================

Explainable AI (XAI) for the SpineLIT vision module — Grad-CAM.

Grad-CAM localises *where* the severity network looked when it decided a
disc's Pfirrmann grade. It backpropagates the logit of a chosen disc/class
to the convolutional feature map of the ConvNeXt backbone and weights each
channel by its average gradient (Selvaraju et al., ICCV 2017):

    alpha_c = global_average_pool(grad( y^class / F^c ))
    L = ReLU( sum_c alpha_c * F^c )   upsampled to the input resolution

The pipeline is:
    severity CNN (ConvNeXt)  ->  ddd_logits(B,5,5)
    GradCAM hook on backbone feature map F
    -> per-disc heatmap overlaying the input MRI

This is intentionally self-contained and testable without a trained
checkpoint: the hooks are architecture-only, so the same code works on a
torch-listed random-weight model and on a fine-tuned checkpoint.

Usage
-----
    python gradcam_demo.py --image dataset/data/processed_spider_jpgs/170_t2.jpg
        --output outputs/demo_gradcam.png
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import NUM_DISCS, NUM_PFRRMANN_CLASSES, PFRRMANN_GRADES, DISC_LEVELS
from model import SpineFoundationModel


class GradCAM:
    """
    Class-activation maps for `SpineFoundationModel.ddd_logits`.

    Hook target: the backbone's *output* — in `features_only` mode the
    backbone returns the stage feature map `(B, C, Hf, Wf)` that the
    disc-localized Pfirrmann head consumes (`model.disc_head`).
    """

    def __init__(self, model: SpineFoundationModel):
        self.model = model
        self._fmap: torch.Tensor | None = None
        self._hook = model.backbone.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        # Keep the LIVE feature tensor (part of the forward graph) so
        # `logit.backward()` populates `.grad` via `retain_grad()`.
        fmap = out[0] if isinstance(out, (tuple, list)) else out
        self._fmap = fmap
        self._fmap.retain_grad()

    def cam_for(self, x: torch.Tensor, disc_index: int,
                class_index: int) -> np.ndarray:
        """
        Feature-map-size Grad-CAM (0..1) for one disc's chosen class.
        `x` is the model input (B,3,H,W), already preprocessed.
        """
        self.model.eval()
        self._fmap = None

        outputs = self.model(x)
        logit = outputs["ddd_logits"][0, disc_index, class_index]

        self.model.zero_grad()
        logit.backward()

        fmap = self._fmap
        grad = fmap.grad
        if grad is None:
            raise RuntimeError("Grad-CAM: no gradient captured for fmap.")

        # (C,) channel importance, then weighted ReLU activation map.
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (fmap * weights).sum(dim=1, keepdim=True).clamp(min=0)

        # Upsample to the spatial size of the input image.
        h, w = x.shape[2], x.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        mx = cam.max()
        return cam / mx if mx > 1e-9 else cam

    def cam_for_top_class(self, x: torch.Tensor,
                          disc_index: int) -> tuple[int, np.ndarray]:
        """Class with the highest logit for a disc, and its Grad-CAM map."""
        with torch.no_grad():
            logits = self.model(x)["ddd_logits"][0, disc_index]
        class_index = int(logits.argmax().item())
        return class_index, self.cam_for(x, disc_index, class_index)

    def close(self) -> None:
        self._hook.remove()


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a 0..1 heatmap to a (H,W,3) jet colour image."""
    import matplotlib.cm as cm

    return (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.55,
) -> Image.Image:
    """Alpha-blend a Grad-CAM heatmap over a (RGB) image."""
    h, w = image.size[1], image.size[0]
    cam_img = Image.fromarray(jet_colormap(cam), "RGB").resize(image.size)
    return Image.blend(image.convert("RGB"), cam_img, alpha)


def explain_discs(
    model: SpineFoundationModel,
    image: Image.Image,
    transform=None,
    disc_index: int | None = None,
) -> dict:
    """
    Convenience: run Grad-CAM for one (or all) disc(s) on a PIL image.

    Returns for each disc `{disc_index, level, grade, class_index,
    heatmap (np.ndarray 0..1), proba}`.
    """
    # Standard preprocessing identical to the vision engine: 512x512 RGB,
    # ImageNet-normalised, channel-first.
    if transform is None:
        transform = _default_transform()
    x = transform(image).unsqueeze(0)

    cam_module = GradCAM(model)

    with torch.no_grad():
        logits = model(x)["ddd_logits"][0]          # (5,5)
        proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    out = {}
    levels = list(DISC_LEVELS)
    for idx in (range(NUM_DISCS) if disc_index is None else [disc_index]):
        class_index = int(proba[idx].argmax())
        cam = cam_module.cam_for(x, idx, class_index)
        out[idx] = {
            "disc_index": idx,
            "level": levels[idx],
            "grade": PFRRMANN_GRADES[class_index],
            "class_index": class_index,
            "proba": float(proba[idx, class_index]),
            "heatmap": cam,
        }

    cam_module.close()
    return out


def _default_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])