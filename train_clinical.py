"""
train_clinical.py
=================

Train the Longitudinal Risk Model from structured patient records.

Expected input: dataset/longitudinal_records.csv

Required columns
----------------
    age, sex, pain_scale, modality, pain_years, start_year, years_ahead,
    future_grade_L1/L2 ... future_grade_L5/S1        (DDD grade 0-4)

Optional columns
----------------
    filename                       -> adds the image embedding as input
    baseline_grade_L1/L2 ...       -> enables progression-risk supervision

Usage
-----
python train_clinical.py
python train_clinical.py --epochs 200
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    BEST_MODEL,
    DISC_LEVELS,
    DEVICE,
    LONGITUDINAL_MODEL_NAME,
)
from clinical_model import ClinicalInputs, LongitudinalRiskModel, encode_clinical


# ---------------------------------------------------------
# Record dataset
# ---------------------------------------------------------

class LongitudinalRecordDataset(Dataset):

    def __init__(self, csv_path: Path, image_features: dict | None = None):
        self.df = pd.read_csv(csv_path)
        self.image_features = image_features or {}

        required = [
            "age", "sex", "pain_scale", "modality",
            "pain_years", "start_year", "years_ahead",
            *[f"future_grade_{level}" for level in DISC_LEVELS],
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing}"
            )

        self.has_baseline = all(
            f"baseline_grade_{level}" in self.df.columns
            for level in DISC_LEVELS
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        clinical = encode_clinical(
            ClinicalInputs(
                age=int(row["age"]),
                sex=str(row["sex"]),
                pain_scale=float(row["pain_scale"]),
                modality=str(row["modality"]),
                pain_years=float(row["pain_years"]),
                start_year=int(row["start_year"]),
            )
        )

        years_ahead = torch.tensor(
            [float(row["years_ahead"])], dtype=torch.float32
        )

        future_grades = torch.tensor(
            [float(row[f"future_grade_{level}"]) / 4.0 for level in DISC_LEVELS],
            dtype=torch.float32,
        )

        features = self.image_features.get(str(row.get("filename", "")))

        if self.has_baseline:
            baseline = torch.tensor(
                [float(row[f"baseline_grade_{level}"]) for level in DISC_LEVELS],
                dtype=torch.float32,
            )
            worsened = (
                (future_grades * 4.0 - baseline) >= 1.0
            ).float()
        else:
            baseline = torch.zeros(5)
            worsened = torch.full((5,), float("nan"))

        return {
            "clinical": clinical,
            "years_ahead": years_ahead,
            "future_grades": future_grades,
            "worsened": worsened,
            "image_features": features,
        }


def _collate(batch):
    stack_images = all(b["image_features"] is not None for b in batch)
    return {
        "clinical": torch.stack([b["clinical"] for b in batch]),
        "years_ahead": torch.cat([b["years_ahead"] for b in batch]),
        "future_grades": torch.stack([b["future_grades"] for b in batch]),
        "worsened": torch.stack([b["worsened"] for b in batch]),
        "image_features": (
            torch.stack([b["image_features"] for b in batch])
            if stack_images else None
        ),
    }


def extract_image_features(records_csv: Path) -> dict:
    """Compute frozen foundation-model embeddings for every referenced image."""
    try:
        from utils import load_model, predict, preprocess_image
        from PIL import Image
    except ImportError:
        return {}

    df = pd.read_csv(records_csv)

    if "filename" not in df.columns:
        return {}

    filenames = [str(f) for f in df["filename"].dropna().unique()]

    try:
        model, device, _ = load_model(BEST_MODEL)
    except FileNotFoundError:
        print("No image checkpoint found -> training clinical-only model.")
        return {}

    features = {}
    print(f"Extracting image embeddings for {len(filenames)} image(s)...")
    for name in filenames:
        matches = list(Path("dataset/data").rglob(name))
        if not matches:
            continue
        image = Image.open(matches[0]).convert("RGB")
        outputs = predict(model, preprocess_image(image), device)
        features[name] = outputs["features"][0].float().cpu()
    return features


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

def train(args) -> None:

    records_csv = Path(args.records)

    if not records_csv.exists():
        print(
            f"No longitudinal records found at {records_csv}."
            "\nCreate dataset/longitudinal_records.csv (schema in README)"
            " and re-run."
        )
        return

    image_features = extract_image_features(records_csv)

    dataset = LongitudinalRecordDataset(records_csv, image_features)
    print(f"Records : {len(dataset)}"
          f" | image-fused : {bool(image_features)}"
          f" | progression labels : {dataset.has_baseline}")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(val_set, batch_size=val_size, collate_fn=_collate)

    model = LongitudinalRiskModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_loss = float("inf")

    for epoch in range(args.epochs):

        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch["clinical"].to(DEVICE),
                batch["image_features"].to(DEVICE)
                if batch["image_features"] is not None else None,
                batch["years_ahead"].mean().item(),
            )

            target_grades = batch["future_grades"].to(DEVICE)

            grade_loss = torch.nn.functional.smooth_l1_loss(
                outputs["future_grade"], target_grades
            )

            worsened = batch["worsened"]
            if not torch.isnan(worsened).all():
                valid = ~torch.isnan(worsened)
                risk_target = worsened.to(DEVICE)[valid].clamp(0, 1)
                risk_loss = torch.nn.functional.binary_cross_entropy(
                    outputs["progression_risk"][valid],
                    risk_target.clamp(min=0.01, max=0.99),
                )
                overall_target = risk_target.mean().expand_as(
                    outputs["overall_risk"]
                )
                overall_loss = torch.nn.functional.mse_loss(
                    outputs["overall_risk"], overall_target
                )
            else:
                risk_loss = torch.tensor(0.0, device=DEVICE)
                overall_loss = torch.tensor(0.0, device=DEVICE)

            loss = grade_loss + risk_loss + 0.5 * overall_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in val_loader:
                outputs = model(
                    batch["clinical"].to(DEVICE),
                    batch["image_features"].to(DEVICE)
                    if batch["image_features"] is not None else None,
                    batch["years_ahead"].mean().item(),
                )
                val_losses.append(
                    torch.nn.functional.smooth_l1_loss(
                        outputs["future_grade"],
                        batch["future_grades"].to(DEVICE),
                    ).item()
                )

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:>3}/{args.epochs}"
                f" | Train {train_loss:.4f} | Val {val_loss:.4f}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_fused": bool(image_features),
                    "best_val_loss": best_loss,
                },
                LONGITUDINAL_MODEL_NAME,
            )

    print(f"\nLongitudinal model saved to : {LONGITUDINAL_MODEL_NAME}")
    print(f"Best validation loss        : {best_loss:.4f}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Train the longitudinal clinical risk model."
    )
    parser.add_argument("--records", default="dataset/longitudinal_records.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
