"""
model.py
========

Spine Foundation Model — multi-task architecture.

Tasks
-----
1. Landmark localization : 10 points (L1-L5 vertebral centres + 5 disc
   centres) with per-point self-supervised confidence.
2. Disc degeneration     : per-level continuous grade 0-4 + confidence.
3. Spondylolisthesis     : per-level slip percent (0-100) + confidence.

Outputs (forward)
-----------------
{
    "coords":              Tensor(B,20),
    "localization_conf":   Tensor(B,10),
    "ddd_grade":           Tensor(B,5),   # sigmoid -> multiply by 4
    "ddd_conf":            Tensor(B,5),
    "spondy_slip":         Tensor(B,5),   # sigmoid -> multiply by 100 (%)
    "spondy_conf":         Tensor(B,5),
    "features":            Tensor(B,512), # shared embedding for clinical model
}
"""

import torch
import torch.nn as nn
import timm

from config import PRETRAINED


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
            nn.Linear(128, 10),
        )

        self.ddd_grade_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 5),
        )

        self.ddd_confidence_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

        self.spondy_slip_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 5),
        )

        self.spondy_confidence_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):

        features = self.shared(self.backbone(x))

        coords = torch.sigmoid(self.coordinate_head(features))

        localization_conf = torch.sigmoid(
            self.localization_confidence_head(features)
        )

        ddd_grade = torch.sigmoid(self.ddd_grade_head(features))

        ddd_conf = torch.sigmoid(self.ddd_confidence_head(features))

        spondy_slip = torch.sigmoid(self.spondy_slip_head(features))

        spondy_conf = torch.sigmoid(self.spondy_confidence_head(features))

        return {
            "coords": coords,
            "localization_conf": localization_conf,
            "ddd_grade": ddd_grade,
            "ddd_conf": ddd_conf,
            "spondy_slip": spondy_slip,
            "spondy_conf": spondy_conf,
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
