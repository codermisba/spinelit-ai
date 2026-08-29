"""
model.py
========

Spine Foundation Model — DDD-only multi-task architecture.

Focus
-----
Disc Degenerative Disease (DDD) only, graded with the Pfirrmann scale
(grades I..V) per disc level on sagittal T2 lumbar MRI.

Tasks
-----
1. Landmark localization : 10 points (L1-L5 vertebral centres + 5 disc
   centres) with per-point self-supervised confidence.
2. Disc degeneration     : per-level Pfirrmann grade (5-class
   classification, I..V). Each disc is classified from a *disc-localized*
   feature vector obtained by Gaussian-attention pooling of the spatial
   feature map around the disc centre predicted by the coordinate head.
   Localizing per-disc features (instead of a single global embedding)
   substantially improves grading accuracy.

Outputs (forward)
-----------------
{
    "coords":              Tensor(B,20),
    "localization_conf":   Tensor(B,10),
    "ddd_logits":          Tensor(B,5,5),  # per-disc x Pfirrmann class logits
    "features":            Tensor(B,512),  # shared embedding (for longitudinal)
}
"""

import torch
import torch.nn as nn
import timm

from config import (
    DISC_FEATURE_STAGE,
    NUM_DISCS,
    NUM_KEYPOINTS,
    NUM_PFRRMANN_CLASSES,
    PRETRAINED,
)


class _DiscLocalizedHead(nn.Module):
    """
    Classify each disc from a Gaussian-attention-pooled feature patch.

    Given the spatial backbone feature map F (B,C,Hf,Wf) and the predicted
    normalized disc centres (B,5,2) (x,y in [0,1] in the input image), build
    a 2D Gaussian attention mask around each centre, pool F -> (B,5,C) and
    run a shared classifier -> (B,5,5) Pfirrmann logits.
    """

    def __init__(self, feature_dim: int, sigma_fraction: float = 0.5):
        super().__init__()
        self.sigma_fraction = sigma_fraction
        # pooled is (B, num_discs, feature_dim) -> (B, num_discs, classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, NUM_PFRRMANN_CLASSES),
        )

    def forward(self, fmap: torch.Tensor, disc_centers: torch.Tensor):
        B, C, Hf, Wf = fmap.shape
        # disc_centers: (B,5,2) normalized (x, y) in [0,1] -> feature-map px
        fx = disc_centers[:, :, 0].clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1) * (Wf - 1)
        fy = disc_centers[:, :, 1].clamp(0.0, 1.0).unsqueeze(-1).unsqueeze(-1) * (Hf - 1)
        yy = torch.arange(Hf, device=fmap.device, dtype=fmap.dtype).view(1, 1, Hf, 1)
        xx = torch.arange(Wf, device=fmap.device, dtype=fmap.dtype).view(1, 1, 1, Wf)
        d2 = (xx - fx).square() + (yy - fy).square()          # (B,5,Hf,Wf)
        sigma = self.sigma_fraction * max(Hf, Wf)
        att = torch.exp(-d2 / (2.0 * sigma * sigma))
        att = att / att.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        pooled = torch.einsum("bchw,bdwh->bdc", fmap, att)    # (B,5,C)
        return self.classifier(pooled)                        # (B,5,5)


class SpineFoundationModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Spatial feature backbone (keeps a mid-stage feature map for
        # disc-localized pooling). Stage index + global average pooling.
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=PRETRAINED,
            features_only=True,
            out_indices=(DISC_FEATURE_STAGE,),
        )
        # Infer the spatial-channel dim of the selected stage.
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 32, 32))
        feature_dim = dummy[0].shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.30),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
        )

        self.coordinate_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 20),
        )

        self.localization_confidence_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, NUM_KEYPOINTS),
        )

        self.disc_head = _DiscLocalizedHead(feature_dim)

    def forward(self, x):

        fmap = self.backbone(x)[0]                     # (B,C,Hf,Wf)
        g = self.pool(fmap).flatten(1)                 # (B,C)
        features = self.shared(g)                      # (B,512)

        coords = torch.sigmoid(self.coordinate_head(features))
        localization_conf = torch.sigmoid(
            self.localization_confidence_head(features)
        )

        # Disc centres = coords[10:20] (after 5 vertebra points), (x, y).
        disc_centers = coords[:, 10:20].view(-1, NUM_DISCS, 2)
        ddd_logits = self.disc_head(fmap, disc_centers)

        return {
            "coords": coords,
            "localization_conf": localization_conf,
            "ddd_logits": ddd_logits,
            "features": features,
        }

    @torch.no_grad()
    def extract_features(self, x) -> torch.Tensor:
        """Shared 512-d image embedding used by the longitudinal model."""
        self.eval()
        return self.forward(x)["features"]


if __name__ == "__main__":

    model = SpineFoundationModel()

    x = torch.randn(2, 3, 256, 256)

    outputs = model(x)

    print()

    for name, value in outputs.items():
        print(f"{name:20s} {tuple(value.shape)}")
