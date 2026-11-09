"""
model.py
========

Spine Foundation Model — spatial multi-task architecture (DDD + anatomy).

Why this file was rebuilt
-------------------------
The previous version pooled the backbone with **global average pooling**
before regressing the 10 landmarks.  Global pooling throws away *where*
things are in the image, so the coordinate head could only learn the
population-average spine position — the "boxes" landed in the wrong place
for individual patients.

This version keeps a dense feature map and predicts a **heatmap per
keypoint** (the standard, reliable approach for landmark localisation,
as used by HRNet / stacked-hourglass / YOLO-pose).  Coordinates are
obtained with a differentiable soft-argmax.  A Feature Pyramid Network
(FPN) fuses ConvNeXt stages so both small discs and large vertebrae are
localised accurately.

Tasks
-----
1. Landmark localisation : 10 points (L1-L5 vertebral centres + 5 disc
   centres), one Gaussian heatmap per point + per-point confidence.
2. Disc degeneration     : per-level Pfirrmann grade (5-class
   classification) from a Gaussian-attention-pooled disc patch.
3. Segmentation          : optional 4-class mask (background / vertebra /
   disc / spinal canal) from the same dense features.  Trained only when
   masks are available; ignored otherwise.

Outputs (forward)
-----------------
{
    "coords":            Tensor(B,20),           # x,y per keypoint in [0,1]
    "localization_conf": Tensor(B,10),           # 0..1 per keypoint
    "ddd_logits":        Tensor(B,5,5),          # disc x Pfirrmann logits
    "features":          Tensor(B,512),          # embedding (longitudinal)
    "heatmaps":          Tensor(B,10,Hf,Wf),     # dense keypoint heatmaps
    "seg_logits":        Tensor(B,4,Hf,Wf),      # optional segmentation
}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config as _cfg


# ----------------------------------------------------------------------
# Feature Pyramid Network
# ----------------------------------------------------------------------

class FPN(nn.Module):
    """
    Lightweight top-down FPN over the ConvNeXt stage outputs.

    Returns a single dense feature map at the highest resolution stage
    (stride 4 for ConvNeXt) so landmark heatmaps are sharp.
    """

    def __init__(self, in_channels: list[int], out_dim: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_dim, kernel_size=1) for c in in_channels
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
        out = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            out = F.interpolate(
                out, size=laterals[i].shape[-2:], mode="nearest"
            )
            out = laterals[i] + out
        return self.smooth(out)


# ----------------------------------------------------------------------
# Helper heads
# ----------------------------------------------------------------------

def _soft_argmax_2d(
    heatmaps: torch.Tensor,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable peak extraction.

    Parameters
    ----------
    heatmaps : (B, K, H, W) logits
    temperature : softmax sharpening factor.  Background logits sit near 0
        while a trained gaussian peak is only ~1-2, so a plain softmax is
        almost uniform and the expectation collapses to the image centre.
        Multiplying the logits by a temperature > 1 makes the spatial
        expectation actually read the peak.

    Returns
    -------
    coords : (B, K*2) normalized (x, y) in [0,1]
    peak   : (B, K)   peak probability (sharpened softmax max) in [0,1]
    """
    if temperature is None:
        temperature = float(getattr(_cfg, "SOFTARGMAX_TEMPERATURE", 1.0))
    b, k, h, w = heatmaps.shape
    logits = heatmaps * float(temperature)
    prob = torch.softmax(logits.flatten(2), dim=-1)            # (B,K,H*W)
    xs = (torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)
          + 0.5) / w
    ys = (torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
          + 0.5) / h
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    x = (prob * grid_x).sum(dim=-1)
    y = (prob * grid_y).sum(dim=-1)
    coords = torch.stack([x, y], dim=-1).reshape(b, k * 2)
    peak = prob.max(dim=-1).values
    return coords, peak


class _DiscLocalizedHead(nn.Module):
    """
    Classify each disc from a Gaussian-attention-pooled feature patch.

    Given the dense FPN feature map F (B,C,Hf,Wf) and the disc centres
    (B,5,2) in normalized coordinates, build a 2-D Gaussian attention
    mask around each centre, pool F -> (B,5,C) and run a shared
    classifier -> (B,5,5) Pfirrmann logits.
    """

    def __init__(self, feature_dim: int, num_discs: int,
                 num_classes: int, sigma_fraction: float = 0.55):
        super().__init__()
        self.num_discs = int(num_discs)
        self.num_classes = int(num_classes)
        self.sigma_fraction = float(sigma_fraction)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, fmap: torch.Tensor, disc_centers: torch.Tensor):
        b, c, hf, wf = fmap.shape
        if disc_centers is None:
            raise ValueError("disc_centers is required")
        disc_centers = disc_centers.reshape(b, self.num_discs, 2)

        fx = disc_centers[:, :, 0].clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        fy = disc_centers[:, :, 1].clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1)
        fx = fx * (wf - 1)
        fy = fy * (hf - 1)

        yy = torch.arange(hf, device=fmap.device, dtype=fmap.dtype).view(
            1, 1, hf, 1)
        xx = torch.arange(wf, device=fmap.device, dtype=fmap.dtype).view(
            1, 1, 1, wf)
        d2 = (xx - fx).square() + (yy - fy).square()          # (B,5,Hf,Wf)

        sigma = self.sigma_fraction * max(hf, wf)
        att = torch.exp(-d2 / (2.0 * sigma * sigma))
        att = att / att.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)

        # fmap: (B,C,H,W), att: (B,D,H,W) -> (B,D,C)
        pooled = torch.einsum("bchw,bdhw->bdc", fmap, att)
        return self.classifier(pooled)                        # (B,5,classes)


# ----------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------

class SpineFoundationModel(nn.Module):

    def __init__(
        self,
        model_name: str | None = None,
        num_keypoints: int | None = None,
        num_discs: int | None = None,
        num_pfrrmann_classes: int | None = None,
        num_seg_classes: int | None = None,
        fpn_dim: int | None = None,
        pretrained: bool | None = None,
        disc_feature_stage: int | None = None,
    ):
        super().__init__()

        # Config values are read at __init__ time (not import time) so that
        # `utils.load_model` can apply per-checkpoint metadata before
        # constructing the network.
        self.model_name = model_name or _cfg.MODEL_NAME
        self.num_keypoints = int(num_keypoints or _cfg.NUM_KEYPOINTS)
        self.num_discs = int(num_discs or _cfg.NUM_DISCS)
        self.num_pfrrmann_classes = int(
            num_pfrrmann_classes or _cfg.NUM_PFRRMANN_CLASSES)
        self.num_seg_classes = int(num_seg_classes or _cfg.NUM_SEG_CLASSES)
        self.fpn_dim = int(fpn_dim or _cfg.FPN_DIM)
        self.disc_feature_stage = int(
            disc_feature_stage or _cfg.DISC_FEATURE_STAGE)

        if pretrained is None:
            pretrained = _cfg.PRETRAINED

        # Spatial feature backbone.  Keep all stages for the FPN.
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            features_only=True,
        )
        channels = list(self.backbone.feature_info.channels())
        # Some small timm models expose fewer stages; use what exists.
        self.fpn = FPN(channels, out_dim=self.fpn_dim)
        fmap_channels = self.fpn_dim

        # Keypoint heatmap head (one channel per landmark).
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(fmap_channels, fmap_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fmap_channels),
            nn.GELU(),
            nn.Conv2d(fmap_channels, self.num_keypoints, 1),
        )
        # Start with a low foreground prior so BCE does not explode.
        nn.init.constant_(self.keypoint_head[-1].bias, -3.0)

        # Segmentation head (optional task).
        self.seg_head = nn.Sequential(
            nn.Conv2d(fmap_channels, fmap_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fmap_channels),
            nn.GELU(),
            nn.Conv2d(fmap_channels, self.num_seg_classes, 1),
        )

        # Global embedding for confidence + longitudinal model.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.BatchNorm1d(fmap_channels),
            nn.Dropout(0.30),
            nn.Linear(fmap_channels, 512),
            nn.GELU(),
        )

        self.localization_confidence_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, self.num_keypoints),
        )

        self.disc_head = _DiscLocalizedHead(
            fmap_channels, self.num_discs, self.num_pfrrmann_classes,
            sigma_fraction=float(getattr(_cfg, "DISC_POOL_SIGMA_FRACTION", 0.12)),
        )

        # Soft-argmax temperature is resolved at construction time so that a
        # checkpoint can pin the value it was trained with via metadata.
        self.softargmax_temperature = float(
            getattr(_cfg, "SOFTARGMAX_TEMPERATURE", 1.0)
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        disc_centers: torch.Tensor | None = None,
        return_seg: bool = True,
    ) -> dict:
        feats = self.backbone(x)                     # list of stage maps
        fmap = self.fpn(feats)                       # (B,C,Hf,Wf) stride 4
        b, c, hf, wf = fmap.shape

        keypoint_logits = self.keypoint_head(fmap)   # (B,K,Hf,Wf)
        coords, peak = _soft_argmax_2d(
            keypoint_logits, self.softargmax_temperature
        )

        g = self.pool(fmap).flatten(1)
        features = self.shared(g)

        localization_conf = torch.sigmoid(
            self.localization_confidence_head(features)
        )

        # Disc pooling uses ground-truth centres during training when
        # supplied (teacher forcing) and predicted centres at inference.
        if disc_centers is None:
            disc_centers = coords[:, self.num_discs * 2:].reshape(
                b, self.num_discs, 2)
        else:
            disc_centers = disc_centers.reshape(b, self.num_discs, 2)
        ddd_logits = self.disc_head(fmap, disc_centers)

        out = {
            "coords": coords,
            "localization_conf": localization_conf,
            "ddd_logits": ddd_logits,
            "features": features,
            "heatmaps": keypoint_logits,
            "heatmap_peak": peak,
            "fmap": fmap,
        }
        if return_seg:
            out["seg_logits"] = self.seg_head(fmap)
        return out

    @torch.no_grad()
    def extract_features(self, x) -> torch.Tensor:
        """Shared 512-d image embedding used by the longitudinal model."""
        self.eval()
        return self.forward(x, return_seg=False)["features"]


if __name__ == "__main__":

    model = SpineFoundationModel()
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)

    print()
    for name, value in outputs.items():
        if torch.is_tensor(value):
            print(f"{name:20s} {tuple(value.shape)}")
