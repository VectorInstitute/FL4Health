import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Rxrx1Dataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        root: Path,
        dataset_type: str,
        transform: Callable | None = None,
        cache_images: bool = False,
    ):
        """
        Args:
            metadata (DataFrame): A DataFrame containing image metadata.
            root (str): Root directory for the image data.
            dataset_type (str): 'train' or 'test' to specify dataset type.
            transform (callable, optional): Optional transformation to apply to the images.
        """
        self.metadata = metadata[metadata["dataset"] == dataset_type]
        self.root = root
        self.transform = transform if transform else None

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.metadata["sirna_id"].unique()))}
        self.metadata["mapped_label"] = self.metadata["sirna_id"].map(self.label_map)

        if cache_images:
            self.images = [self.load_image(dict(row)) for _, row in self.metadata.iterrows()]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = dict(self.metadata.iloc[idx])

        if hasattr(self, "images"):
            image = self.images[idx]
        else:
            image = self.load_image(row)
        label = row["mapped_label"]

        return image, label

    def load_image(self, row: dict[str, Any]) -> torch.Tensor:
        experiment = row["experiment"]
        plate = row["plate"]
        well = row["well"]
        site = row["site"]

        images = []
        for channel in range(1, 4):
            image_path = os.path.join(self.root, f"images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png")
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found at {image_path}")
            image = ToTensor(Image.open(image_path).convert("L"))
            images.append(image)
        return torch.cat(images, dim=0)
