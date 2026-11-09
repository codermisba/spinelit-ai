"""
explainability.py
=================

Explainable AI (XAI) for the spine vision module — Grad-CAM.

Grad-CAM localises *where* the network looked when it assigned a disc's
Pfirrmann grade.  It backpropagates the chosen class logit to the dense
feature map that the disc head actually consumes and weights each channel
by its average gradient (Selvaraju et al., ICCV 2017):

    alpha_c = global_average_pool(grad(y^class) / grad(F^c))
    L = ReLU( sum_c alpha_c * F^c )   upsampled to the input resolution

Because the rebuilt model classifies each disc from the **FPN feature map**
(with Gaussian attention around the predicted disc centre), the hook is
placed on ``model.fpn`` — not on the backbone, whose output is a list of
multi-scale maps.  This keeps the explanation aligned with the tensor the
classifier really uses.

Two modes
---------
- ``cam_for`` / ``cam_for_top_class``: a **grade** explanation (needs a
  trained DDD head).
- ``cam_for_localization``: a **localization** explanation for one landmark
  heatmap (works even without Pfirrmann labels).  Callers must label it
  ``kind="localization"``; it is never a grade explanation.

The hooks are architecture-only, so the same code works on a randomly
initialised model (CI/tests) and on a fine-tuned checkpoint.

Usage
-----
    python gradcam_demo.py --image dataset/data/processed_spider_jpgs/170_t2.jpg \
        --output outputs/demo_gradcam.png
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import DISC_LEVELS, NUM_DISCS, PFRRMANN_GRADES


class GradCAM:
    """
    Class-activation maps for ``SpineFoundationModel.ddd_logits``.

    Hook target: the FPN fused feature map ``(B, C, Hf, Wf)`` at stride 4,
    i.e. exactly the tensor the disc-localized Pfirrmann head pools.
    """

    def __init__(self, model, target_layer: str = "fpn"):
        self.model = model
        self._fmap: torch.Tensor | None = None

        if target_layer == "fpn" and hasattr(model, "fpn"):
            module = model.fpn
        else:
            module = model.backbone
        self._module = module
        self._hook = module.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        fmap = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(fmap) or fmap.dim() != 4:
            return
        # Under `torch.no_grad()` (e.g. reading the top class) the tensor has
        # no grad; do not call retain_grad and do not replace a grad-carrying
        # capture from the explainability forward pass.
        if fmap.requires_grad:
            fmap.retain_grad()
            self._fmap = fmap

    def _forward(self, x: torch.Tensor):
        """Forward pass that keeps the image dtype in sync with the model."""
        model_dtype = next(self.model.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        return self.model(x, return_seg=False)

    def _activation_map(self, fmap: torch.Tensor, x: torch.Tensor) -> np.ndarray:
        """Turn the captured feature map + its gradient into a 0..1 CAM."""
        grad = fmap.grad
        if grad is None:
            raise RuntimeError("Grad-CAM: no gradient captured for fmap.")
        weights = grad.float().mean(dim=(2, 3), keepdim=True)
        cam = (fmap.float() * weights).sum(dim=1, keepdim=True).clamp(min=0)
        h, w = x.shape[2], x.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        mx = float(cam.max())
        return cam / mx if mx > 1e-9 else cam

    def cam_for(self, x: torch.Tensor, disc_index: int,
                class_index: int) -> np.ndarray:
        """
        Feature-map-size Grad-CAM (0..1) for one disc's chosen class
        (a *grade* explanation — only meaningful with a trained DDD head).
        ``x`` is the model input (B,3,H,W), already preprocessed.
        """
        self.model.eval()
        self._fmap = None

        outputs = self._forward(x)
        logit = outputs["ddd_logits"][0, disc_index, class_index]

        self.model.zero_grad(set_to_none=True)
        logit.backward()

        fmap = self._fmap
        if fmap is None:
            raise RuntimeError("Grad-CAM: feature map was not captured.")
        return self._activation_map(fmap, x)

    def cam_for_localization(self, x: torch.Tensor,
                             keypoint_index: int) -> np.ndarray:
        """
        Grad-CAM (0..1) for a *localization* decision: backpropagates the
        strongest heatmap logit of one keypoint to the fused FPN map.

        This explains where the localizer looked for that landmark.  It is
        NOT a Pfirrmann-grade explanation and is labelled ``kind=localization``
        by callers so the two can never be confused.
        """
        self.model.eval()
        self._fmap = None

        outputs = self._forward(x)
        heat = outputs["heatmaps"][0, keypoint_index]
        score = heat.flatten(1).max(dim=-1).values.sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        fmap = self._fmap
        if fmap is None:
            raise RuntimeError("Grad-CAM: feature map was not captured.")
        return self._activation_map(fmap, x)

    @torch.no_grad()
    def top_class(self, x: torch.Tensor, disc_index: int) -> int:
        """Class with the highest logit for a disc."""
        logits = self._forward(x)["ddd_logits"][0, disc_index]
        return int(logits.argmax().item())

    def cam_for_top_class(self, x: torch.Tensor,
                          disc_index: int) -> tuple[int, np.ndarray]:
        """Class with the highest logit for a disc, and its Grad-CAM map."""
        class_index = self.top_class(x, disc_index)
        return class_index, self.cam_for(x, disc_index, class_index)

    def close(self) -> None:
        try:
            self._hook.remove()
        except Exception:  # noqa: BLE001 - idempotent close
            pass


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a 0..1 heatmap to a (H,W,3) jet colour image."""
    import matplotlib.cm as cm

    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    return (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.55,
) -> Image.Image:
    """Alpha-blend a Grad-CAM heatmap over a (RGB) image."""
    cam_img = Image.fromarray(jet_colormap(cam), "RGB").resize(image.size)
    return Image.blend(image.convert("RGB"), cam_img, float(alpha))


def explain_discs(
    model,
    image: Image.Image,
    transform=None,
    disc_index: int | None = None,
    device=None,
) -> dict:
    """
    Convenience: run Grad-CAM for one (or all) disc(s) on a PIL image.

    Returns for each disc ``{disc_index, level, grade, class_index,
    heatmap (np.ndarray 0..1), proba}``.
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocessing identical to the vision engine (config image size,
    # ImageNet normalisation).
    if transform is None:
        from utils import get_transform
        transform = get_transform()
    x = transform(image).unsqueeze(0).to(device)

    cam_module = GradCAM(model)
    try:
        with torch.no_grad():
            outputs = model(x, return_seg=False)
            proba = torch.softmax(
                outputs["ddd_logits"][0].float(), dim=-1
            ).cpu().numpy()

        out = {}
        levels = list(DISC_LEVELS)
        indices = range(NUM_DISCS) if disc_index is None else [disc_index]
        for idx in indices:
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
    finally:
        cam_module.close()
    return out
