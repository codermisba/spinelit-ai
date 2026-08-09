
"""
dataset.py
==========

Spine Foundation Dataset

Returns
-------
{
    "image": Tensor(3,H,W),
    "coords": Tensor(10),
    "confidence": Tensor(5)
}
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import DATA_DIR, IMAGE_SIZE


class SpineDataset(Dataset):

    def __init__(
        self,
        annotation_file="dataset/coords_pretrain.csv",
        transform=None,
    ):

        self.annotation_file = Path(annotation_file)

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                ]
            )

        self.transform = transform

        self.df = pd.read_csv(self.annotation_file)

        self.df["level"] = self.df["level"].astype(str).str.strip()

        self.level_order = [
            "L1/L2",
            "L2/L3",
            "L3/L4",
            "L4/L5",
            "L5/S1",
        ]

        self.groups = self.df.groupby("filename")

        self.image_names = list(self.groups.groups.keys())

        self.image_dirs = [
            DATA_DIR / "processed_lsd_jpgs",
            DATA_DIR / "processed_osf_jpgs",
            DATA_DIR / "processed_spider_jpgs",
            DATA_DIR / "processed_tseg_jpgs",
        ]

    def __len__(self):
        return len(self.image_names)

    def _find_image(self, filename):

        for folder in self.image_dirs:

            image_path = folder / filename

            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"Image not found: {filename}")

    def __getitem__(self, index):

        filename = self.image_names[index]

        rows = self.groups.get_group(filename).copy()

        rows["level"] = rows["level"].astype(str).str.strip()

        rows = rows.set_index("level")

        keypoints = []

        for level in self.level_order:

            if level in rows.index:

                row = rows.loc[level]

                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                keypoints.extend(
                    [
                        float(row["relative_x"]),
                        float(row["relative_y"]),
                    ]
                )

            else:

                keypoints.extend([0.0, 0.0])

        coords = torch.tensor(keypoints, dtype=torch.float32)

        confidence = torch.ones(5, dtype=torch.float32)

        image_path = self._find_image(filename)

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return {
            "image": image,
            "coords": coords,
            "confidence": confidence,
        }


if __name__ == "__main__":

    dataset = SpineDataset()

    print(f"Dataset Size : {len(dataset)}")

    sample = dataset[0]

    print()

    print("Image")

    print(sample["image"].shape)

    print()

    print("Coordinates")

    print(sample["coords"])

    print()

    print("Confidence")

    print(sample["confidence"])

