
"""
model.py
========

Spine Foundation Model

Outputs
-------
{
    "coords": Tensor(B,10),
    "confidence": Tensor(B,5),
    "features": Tensor(B,512)
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

            nn.Linear(256, 10),
        )

        self.confidence_head = nn.Sequential(

            nn.Dropout(0.25),

            nn.Linear(512, 128),

            nn.GELU(),

            nn.Linear(128, 5),
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.shared(features)

        coords = torch.sigmoid(
            self.coordinate_head(features)
        )

        confidence = torch.sigmoid(
            self.confidence_head(features)
        )

        return {
            "coords": coords,
            "confidence": confidence,
            "features": features,
        }


if __name__ == "__main__":

    model = SpineFoundationModel()

    x = torch.randn(2, 3, 512, 512)

    outputs = model(x)

    print()

    print("Coordinates")

    print(outputs["coords"].shape)

    print()

    print("Confidence")

    print(outputs["confidence"].shape)

    print()

    print("Features")

    print(outputs["features"].shape)
