"""
clinical_model.py
=================

Longitudinal Spine Intelligence — clinical + imaging fusion model.

Fuses structured patient data with the foundation model's image embedding
to predict FUTURE disc-degeneration state:

Inputs (clinical)
-----------------
    age          : years
    sex          : "male" | "female"
    pain_scale   : VAS 0-10
    modality     : "xray" | "mri"
    pain_years   : years the patient has been in pain
    start_year   : calendar year symptoms started

Conditioning
------------
    years_ahead  : prediction horizon (0-10 years)

Outputs
-------
{
    "future_grade":      Tensor(B,5),  # predicted DDD grade 0-4 at horizon
    "progression_risk":  Tensor(B,5),  # probability of worsening >= 1 grade
    "overall_risk":      Tensor(B,),   # global longitudinal risk 0-1
}
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import (
    AGE_RANGE,
    CLINICAL_FEATURE_DIM,
    HORIZON_RANGE,
    IMAGE_FEATURE_DIM,
    LONGITUDINAL_HIDDEN_DIM,
    MODALITY_CATEGORIES,
    PAIN_YEARS_RANGE,
    SEX_CATEGORIES,
    START_YEAR_RANGE,
    VAS_RANGE,
)


@dataclass
class ClinicalInputs:
    age: int
    sex: str
    pain_scale: float
    modality: str
    pain_years: float
    start_year: int


def _minmax(value: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (float(value) - lo) / max(hi - lo, 1e-6)))


def encode_clinical(inputs: ClinicalInputs) -> torch.Tensor:
    """Normalize raw clinical fields into an 8-d feature vector."""
    sex_onehot = [
        1.0 if inputs.sex.strip().lower() == category else 0.0
        for category in SEX_CATEGORIES
    ]
    modality_onehot = [
        1.0 if inputs.modality.strip().lower() == m else 0.0
        for m in MODALITY_CATEGORIES
    ]
    features = [
        _minmax(inputs.age, *AGE_RANGE),
        *sex_onehot,
        _minmax(inputs.pain_scale, *VAS_RANGE),
        *modality_onehot,
        _minmax(inputs.pain_years, *PAIN_YEARS_RANGE),
        _minmax(inputs.start_year, *START_YEAR_RANGE),
    ]
    return torch.tensor(features, dtype=torch.float32)


class LongitudinalRiskModel(nn.Module):

    def __init__(
        self,
        clinical_dim: int = 8,
        image_dim: int = IMAGE_FEATURE_DIM,
    ):
        super().__init__()

        self.image_dim = image_dim

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, CLINICAL_FEATURE_DIM),
            nn.GELU(),
        )

        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, CLINICAL_FEATURE_DIM),
            nn.GELU(),
        )

        self.horizon_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        fused = 2 * CLINICAL_FEATURE_DIM + 32

        self.trunk = nn.Sequential(
            nn.Linear(fused, LONGITUDINAL_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(LONGITUDINAL_HIDDEN_DIM, 128),
            nn.GELU(),
        )

        self.future_grade_head = nn.Linear(128, 5)
        self.progression_head = nn.Linear(128, 5)
        self.overall_head = nn.Linear(128, 1)

    def forward(
        self,
        clinical_features: torch.Tensor,
        image_features: torch.Tensor | None = None,
        years_ahead: float = 5.0,
    ):
        clinical = self.clinical_encoder(clinical_features)

        if image_features is None:
            image_features = torch.zeros(
                clinical.shape[0], self.image_dim, device=clinical.device,
            )
        image = self.image_projector(image_features)

        horizon = torch.tensor(
            [[_minmax(years_ahead, *HORIZON_RANGE)]],
            device=clinical.device,
        ).expand(clinical.shape[0], 1)
        horizon_embedding = self.horizon_encoder(horizon)

        fused = self.trunk(torch.cat([clinical, image, horizon_embedding], dim=1))

        return {
            "future_grade": torch.sigmoid(self.future_grade_head(fused)),
            "progression_risk": torch.sigmoid(self.progression_head(fused)),
            "overall_risk": torch.sigmoid(self.overall_head(fused)).squeeze(-1),
        }


@torch.no_grad()
def predict_risk(
    model: LongitudinalRiskModel,
    clinical_inputs: ClinicalInputs,
    image_features: torch.Tensor | None = None,
    years_ahead: float = 5.0,
    device: torch.device | None = None,
) -> dict:
    """Run a single-patient longitudinal risk prediction."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    clinical_batch = encode_clinical(clinical_inputs).unsqueeze(0).to(device)
    image_batch = (
        image_features.unsqueeze(0).to(device)
        if image_features is not None
        else None
    )

    outputs = model(clinical_batch, image_batch, years_ahead)

    return {
        "future_grade": (outputs["future_grade"] * 4.0)[0].cpu().tolist(),
        "progression_risk": outputs["progression_risk"][0].cpu().tolist(),
        "overall_risk": float(outputs["overall_risk"][0].cpu()),
        "years_ahead": years_ahead,
    }


if __name__ == "__main__":

    model = LongitudinalRiskModel()

    clinical = encode_clinical(
        ClinicalInputs(58, "female", 6.5, "mri", 4.0, 2022)
    ).unsqueeze(0)

    outputs = model(clinical, torch.randn(1, IMAGE_FEATURE_DIM), 5.0)

    print()

    for name, value in outputs.items():
        print(f"{name:18s} {tuple(value.shape)}")
