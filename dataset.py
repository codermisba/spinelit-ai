"""
dataset.py
==========

Spine Foundation Dataset (DDD-only)

Returns
-------
{
    "image":          Tensor(3,H,W),
    "coords":         Tensor(20),   # [x,y] x 5 vertebrae then x 5 discs (0-1)
    "point_visible":  Tensor(10),   # 1 = supervised point, 0 = masked out
    "ddd_class":      Tensor(5),    # Pfirrmann class index 0-4 per disc (I..V)
    "ddd_mask":       Tensor(5),    # 1 when a Pfirrmann label exists for this level
}
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    DATA_DIR,
    DISC_LANDMARK_CSV,
    DDD_LABELS_CSV,
    DERIVE_VERTEBRA_CENTERS,
    IMAGE_SIZE,
    NUM_DISCS,
    NUM_KEYPOINTS,
    NUM_PFRRMANN_CLASSES,
    NUM_VERTEBRAE,
    VERTEBRA_LANDMARK_CSV,
    VERTEBRAE,
    DISC_LEVELS,
)


def _read_landmark_csv(csv_path: Path, key_column: str) -> pd.DataFrame | None:
    """Load a long-format landmark CSV (filename,key,relative_x,relative_y)."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = {"filename", key_column, "relative_x", "relative_y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns {sorted(required)}"
        )
    df["filename"] = df["filename"].astype(str).str.strip()
    df[key_column] = df[key_column].astype(str).str.strip()
    return df


class SpineDataset(Dataset):

    def __init__(
        self,
        disc_annotation_file=DISC_LANDMARK_CSV,
        vertebra_annotation_file=VERTEBRA_LANDMARK_CSV,
        ddd_labels_file=DDD_LABELS_CSV,
        transform=None,
    ):

        self.disc_csv = _read_landmark_csv(Path(disc_annotation_file), "level")
        if self.disc_csv is None:
            raise FileNotFoundError(
                f"Disc landmark annotations not found: {disc_annotation_file}"
            )

        self.vert_csv = _read_landmark_csv(
            Path(vertebra_annotation_file), "vertebra"
        )

        self.ddd_labels = self._load_grade_labels(
            ddd_labels_file, "level", "pfirrmann_grade"
        )

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                ]
            )
        self.transform = transform

        self.level_order = list(DISC_LEVELS)
        self.vertebra_order = list(VERTEBRAE)

        self.groups = self.disc_csv.groupby("filename")
        self.image_names = list(self.groups.groups.keys())

        self.image_dirs = [
            DATA_DIR / "processed_spider_jpgs",
            DATA_DIR / "processed_lsd_jpgs",
            DATA_DIR / "processed_osf_jpgs",
            DATA_DIR / "processed_tseg_jpgs",
        ]

        # Task availability flag used by train.py to enable/disable the DDD loss
        self.has_ddd_labels = (
            len(set(self.ddd_labels["filename"]) & set(self.image_names)) > 0
            if self.ddd_labels is not None
            else False
        )

    @staticmethod
    def _load_grade_labels(csv_path, key_column, value_column):
        path = Path(csv_path)
        if not path.exists():
            return None
        df = pd.read_csv(path)
        required = {"filename", key_column, value_column}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        df["filename"] = df["filename"].astype(str).str.strip()
        df[key_column] = df[key_column].astype(str).str.strip()
        return df

    def __len__(self):
        return len(self.image_names)

    def _find_image(self, filename):
        for folder in self.image_dirs:
            image_path = folder / filename
            if image_path.exists():
                return image_path
        raise FileNotFoundError(f"Image not found: {filename}")

    # ---------------------------------------------------------
    # Landmark assembly
    # ---------------------------------------------------------

    @staticmethod
    def _lookup(rows, key):
        row = rows.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return float(row["relative_x"]), float(row["relative_y"])

    @staticmethod
    def _derive_vertebra_centers(disc_points: dict) -> dict:
        """
        Weak vertebra-centre supervision from annotated disc centroids.

        L2..L5 = midpoint between adjacent disc centres.
        L1     = one inter-discal spacing extrapolated above L1/L2.
        """
        centers = {}
        ordered = [
            np.array(disc_points[level]) for level in DISC_LEVELS
        ]

        for vertebra_idx in range(len(DISC_LEVELS) - 1):
            upper = ordered[vertebra_idx]
            lower = ordered[vertebra_idx + 1]
            centers[VERTEBRAE[vertebra_idx + 1]] = tuple(
                ((upper + lower) / 2.0).tolist()
            )

        spacings = [
            float(np.linalg.norm(ordered[i] - ordered[i + 1]))
            for i in range(len(ordered) - 1)
        ]
        spacing = float(np.mean(spacings))

        top = ordered[0]
        below = ordered[1]
        direction = top - below
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([0.0, -1.0])
        centers["L1"] = tuple((top + direction * spacing).tolist())

        return centers

    def _get_landmarks(self, filename: str):
        rows = self.groups.get_group(filename).copy()
        rows["level"] = rows["level"].astype(str).str.strip()
        rows = rows.set_index("level")

        disc_points = {}
        disc_visible = []
        for level in self.level_order:
            if level in rows.index:
                disc_points[level] = self._lookup(rows, level)
                disc_visible.append(1.0)
            else:
                disc_points[level] = (0.0, 0.0)
                disc_visible.append(0.0)

        vertebra_points = {v: (0.0, 0.0) for v in self.vertebra_order}
        vertebra_visible = [0.0] * NUM_VERTEBRAE

        if self.vert_csv is not None:
            vrows = self.vert_csv[
                self.vert_csv["filename"] == filename
            ].set_index("vertebra")
            for i, vertebra in enumerate(self.vertebra_order):
                if vertebra in vrows.index:
                    vertebra_points[vertebra] = self._lookup(vrows, vertebra)
                    vertebra_visible[i] = 1.0

        elif DERIVE_VERTEBRA_CENTERS and sum(disc_visible[:4]) >= 3:
            derived = self._derive_vertebra_centers(disc_points)
            for i, vertebra in enumerate(self.vertebra_order):
                vertebra_points[vertebra] = derived[vertebra]
                vertebra_visible[i] = 1.0

        coords = []
        for point in self.vertebra_order:
            coords.extend(vertebra_points[point])
        for level in self.level_order:
            coords.extend(disc_points[level])

        visibility = torch.tensor(
            vertebra_visible + disc_visible, dtype=torch.float32
        )

        return coords, visibility

    # ---------------------------------------------------------
    # Grading labels
    # ---------------------------------------------------------

    @staticmethod
    def _grade_targets(labels_df, filename, key_column, value_column, levels):
        """
        Return (class_targets, mask) where Pfirrmann grade (1-5) in the label
        CSV is converted to a zero-based class index (0-4).
        """
        targets = torch.zeros(len(levels), dtype=torch.long)
        mask = torch.zeros(len(levels), dtype=torch.float32)
        if labels_df is None:
            return targets, mask
        rows = labels_df[labels_df["filename"] == filename].set_index(key_column)
        for i, level in enumerate(levels):
            if level not in rows.index:
                continue
            val = rows.loc[level, value_column]
            # val may be a scalar, or a pandas Series/DataFrame when a level
            # appears in more than one row (duplicate grade annotations).
            if isinstance(val, pd.DataFrame):
                val = val[value_column]
            if isinstance(val, pd.Series):
                vals = pd.to_numeric(val, errors="coerce").dropna()
                if vals.empty:
                    continue
                value = float(vals.median())
            else:
                try:
                    value = float(val)
                except (TypeError, ValueError):
                    continue
            cls = int(round(value)) - 1
            cls = min(max(cls, 0), NUM_PFRRMANN_CLASSES - 1)
            targets[i] = cls
            mask[i] = 1.0
        return targets, mask

    def __getitem__(self, index):

        filename = self.image_names[index]

        coords, visibility = self._get_landmarks(filename)

        ddd_class, ddd_mask = self._grade_targets(
            self.ddd_labels, filename, "level", "pfirrmann_grade",
            self.level_order,
        )

        image_path = self._find_image(filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "point_visible": visibility,
            "ddd_class": ddd_class,
            "ddd_mask": ddd_mask,
        }


if __name__ == "__main__":

    dataset = SpineDataset()

    print(f"Dataset Size : {len(dataset)}")
    print(f"DDD labels available       : {dataset.has_ddd_labels}")

    sample = dataset[0]

    print()
    print("Image")
    print(sample["image"].shape)
    print()
    print("Coordinates")
    print(sample["coords"])
    print()
    print("Point Visibility")
    print(sample["point_visible"])
    print()
    print("DDD class (0-4) & mask")
    print(sample["ddd_class"])
    print(sample["ddd_mask"])
