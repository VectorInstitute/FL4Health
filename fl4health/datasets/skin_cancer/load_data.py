import json
import os
import random
from logging import INFO
from typing import Callable, Dict, Optional, Tuple, Union
from pathlib import Path

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader, RandomSampler

from fl4health.utils.dataset import BaseDataset
from fl4health.datasets.skin_cancer.dataset import SkinCancerDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def load_skin_cancer_data(
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    train_transform: Union[None, Callable] = None,
    test_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load skin cancer dataset (training, validation, and test set)."""
    common_path = "fl4health/datasets/skin_cancer"

    if isinstance(dataset_name, Path):
        dataset_name = dataset_name.name

    dataset_paths = {
        "Barcelona": os.path.join(common_path, "ISIC_2019", "ISIC_19_Barcelona.json"),
        "Rosendahl": os.path.join(common_path, "HAM10000", "HAM_rosendahl.json"),
        "Vienna": os.path.join(common_path, "HAM10000", "HAM_vienna.json"),
        "UFES": os.path.join(common_path, "PAD-UFES-20", "PAD_UFES_20.json"),
        "Canada": os.path.join(common_path, "Derm7pt", "Derm7pt.json"),
    }

    dataset_path = dataset_paths[dataset_name]
    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    random.seed(seed)
    random.shuffle(data)
    total_size = len(data)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]

    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])
        
    if train_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(), 
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            ])

    train_ds: BaseDataset = SkinCancerDataset(train_data, transform=train_transform)
    valid_ds: BaseDataset = SkinCancerDataset(valid_data, transform=test_transform)

    if sampler is not None:
        train_ds = sampler.subsample(train_ds)
        valid_ds = sampler.subsample(valid_ds)

    if dataset_converter is not None:
        train_ds = dataset_converter.convert_dataset(train_ds)
        valid_ds = dataset_converter.convert_dataset(valid_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(valid_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(valid_ds)}

    return train_loader, validation_loader, num_examples


def load_skin_cancer_test_data(
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Union[None, Callable] = None,
    target_transform: Union[None, Callable] = None,
) -> Tuple[DataLoader, Dict[str, int]]:
    """Load skin cancer dataset (test set only)."""
    common_path = "fl4health/datasets"

    dataset_paths = {
        "Barcelona": os.path.join(common_path, "ISIC_2019", "ISIC_19_Barcelona.json"),
        "Rosendahl": os.path.join(common_path, "HAM10000", "HAM_rosendahl.json"),
        "Vienna": os.path.join(common_path, "HAM10000", "HAM_vienna.json"),
        "UFES": os.path.join(common_path, "PAD-UFES-20", "PAD_UFES_20.json"),
        "Canada": os.path.join(common_path, "Derm7pt", "Derm7pt.json"),
    }

    dataset_path = dataset_paths[dataset_name]
    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    random.seed(0)  # For consistency
    random.shuffle(data)
    total_size = len(data)
    test_size = int(0.15 * total_size)
    test_data = data[-test_size:]

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    test_ds: BaseDataset = BaseDataset(test_data, transform=transform, target_transform=target_transform)

    if sampler is not None:
        test_ds = sampler.subsample(test_ds)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_examples = {"test_set": len(test_ds)}
    return test_loader, num_examples
