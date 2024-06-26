import json
import random
from concurrent.futures import ThreadPoolExecutor
from logging import ERROR, INFO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from PIL import Image
from torch.utils.data import DataLoader

from fl4health.utils.dataset import BaseDataset, TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def construct_skin_cancer_tensor_dataset(
    data: List[Dict[str, Any]], transform: Optional[Callable] = None, num_workers: int = 8
) -> TensorDataset:
    def load_image(item: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        image_path = item["img_path"]
        image = Image.open(image_path).convert("RGB")
        target = int(torch.tensor(item["extended_labels"]).argmax().item())
        return image, target

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_image, data))

    data_list, targets_list = zip(*results)
    data_tensor = torch.stack(list(data_list))
    targets_tensor = torch.tensor(list(targets_list))

    return TensorDataset(data_tensor, targets_tensor, transform)


def load_skin_cancer_data(
    data_dir: Path,
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    train_transform: Union[None, Callable] = None,
    val_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load skin cancer dataset (training and validation set)."""

    dataset_paths = {
        "Barcelona": data_dir / "ISIC_2019" / "ISIC_19_Barcelona.json",
        "Rosendahl": data_dir / "HAM10000" / "HAM_rosendahl.json",
        "Vienna": data_dir / "HAM10000" / "HAM_vienna.json",
        "UFES": data_dir / "PAD-UFES-20" / "PAD_UFES_20.json",
        "Canada": data_dir / "Derm7pt" / "Derm7pt.json",
    }

    if dataset_name not in dataset_paths:
        log(ERROR, f"Dataset {dataset_name} not found in available datasets.")
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    dataset_path = dataset_paths[dataset_name]

    if not dataset_path.exists():
        log(
            ERROR,
            f"Dataset file {dataset_path} does not exist.\
            Please follow the instructions in fl4health/datasets/skin_cancer/README.md.",
        )
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

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
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ]
        )

    if val_transform is None:
        val_transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ]
        )

    train_ds: BaseDataset = construct_skin_cancer_tensor_dataset(train_data, transform=train_transform)
    valid_ds: BaseDataset = construct_skin_cancer_tensor_dataset(valid_data, transform=val_transform)

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
    data_dir: Path,
    dataset_name: str,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    test_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    seed: int = 0,
) -> Tuple[DataLoader, Dict[str, int]]:
    """Load skin cancer test dataset."""

    dataset_paths = {
        "Barcelona": data_dir / "ISIC_2019" / "ISIC_19_Barcelona.json",
        "Rosendahl": data_dir / "HAM10000" / "HAM_rosendahl.json",
        "Vienna": data_dir / "HAM10000" / "HAM_vienna.json",
        "UFES": data_dir / "PAD-UFES-20" / "PAD_UFES_20.json",
        "Canada": data_dir / "Derm7pt" / "Derm7pt.json",
    }

    if dataset_name not in dataset_paths:
        log(ERROR, f"Dataset {dataset_name} not found in available datasets.")
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    dataset_path = dataset_paths[dataset_name]

    if not dataset_path.exists():
        log(
            ERROR,
            f"Dataset file {dataset_path} does not exist.\
            Please follow the instructions in fl4health/datasets/skin_cancer/README.md.",
        )
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    random.seed(seed)
    random.shuffle(data)
    total_size = len(data)
    test_size = int(0.15 * total_size)

    test_data = data[-test_size:]

    if test_transform is None:
        test_transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ]
        )

    test_ds: BaseDataset = construct_skin_cancer_tensor_dataset(test_data, transform=test_transform)

    if sampler is not None:
        test_ds = sampler.subsample(test_ds)

    if dataset_converter is not None:
        test_ds = dataset_converter.convert_dataset(test_ds)

    test_loader = DataLoader(test_ds, batch_size=batch_size)

    num_examples = {"test_set": len(test_ds)}
    return test_loader, num_examples
