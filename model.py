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
   classification, I..V) with a class-softmax confidence.

Outputs (forward)
-----------------
{
    "coords":              Tensor(B,20),
    "localization_conf":   Tensor(B,10),
    "ddd_logits":          Tensor(B,5,5),  # per-disc x Pfirrmann class logits
    "features":            Tensor(B,512),  # shared embedding
}
"""

import torch
import torch.nn as nn
import timm

from config import NUM_DISCS, NUM_KEYPOINTS, NUM_PFRRMANN_CLASSES, PRETRAINED


class SpineFoundationModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

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

        self.ddd_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, NUM_DISCS * NUM_PFRRMANN_CLASSES),
        )

    def forward(self, x):

        features = self.shared(self.backbone(x))

        coords = torch.sigmoid(self.coordinate_head(features))

        localization_conf = torch.sigmoid(
            self.localization_confidence_head(features)
        )

        # Reshape to (B, num_discs, num_classes) logits -> softmax in decoder
        ddd_logits = self.ddd_head(features).view(
            -1, NUM_DISCS, NUM_PFRRMANN_CLASSES
        )

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
