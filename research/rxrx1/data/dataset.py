import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Rxrx1Dataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, root: Path, dataset_type: str, transform: Optional[Callable] = None):
        """
        Args:
            metadata (DataFrame): A DataFrame containing image metadata.
            root (str): Root directory for the image data.
            dataset_type (str): 'train' or 'test' to specify dataset type.
            transform (callable, optional): Optional transformation to apply to the images.
        """
        self.metadata = metadata[metadata["dataset"] == dataset_type]
        self.root = root
        self.transform = transform if transform else ToTensor()

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.metadata["sirna_id"].unique()))}
        self.metadata["mapped_label"] = self.metadata["sirna_id"].map(self.label_map)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        experiment = row["experiment"]
        plate = row["plate"]
        well = row["well"]
        site = row["site"]
        label = row["mapped_label"]  # Get the label index

        images = []
        for channel in range(1, 4):
            image_path = os.path.join(self.root, f"images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png")
            image = self.load_image(image_path)
            images.append(image)

        concatenated_image = torch.cat(images, dim=0)
        return concatenated_image, label

    def load_image(self, path: str) -> torch.Tensor:
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found at {path}")
        image = Image.open(path).convert("L")  # Load as grayscale
        return self.transform(image)
