# The following code is adapted from the medical_federated GitHub repository by Seongjun Yang et al.
# Paper: https://arxiv.org/abs/2207.03075
# Code: https://github.com/wns823/medical_federated.git
# - medical_federated/skin_cancer_federated/split_dataset.py
# - medical_federated/skin_cancer_federated/skin_cancer_datasets.py

import json
import random
from concurrent.futures import ThreadPoolExecutor
from logging import INFO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from PIL import Image
from torch.utils.data import DataLoader

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def load_image(item: Dict[str, Any], transform: Optional[Callable]) -> Tuple[torch.Tensor, int]:
    """
    Load and transform an image from a given item dictionary.

    Args:
        item (Dict[str, Any]): A dictionary containing image path and labels.
        transform (Optional[Callable]): Transformation function to apply to the images.

    Returns:
        Tuple[torch.Tensor, int]: A tuple containing the transformed image tensor and the target label.
    """
    image_path = item["img_path"]
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    else:
        # Default transformation if none provided
        image = transforms.ToTensor()(image)
    assert isinstance(image, torch.Tensor), f"Image at {image_path} is not a Tensor"
    target = int(torch.tensor(item["extended_labels"]).argmax().item())
    return image, target


def construct_skin_cancer_tensor_dataset(
    data: List[Dict[str, Any]], transform: Optional[Callable] = None, num_workers: int = 8
) -> TensorDataset:
    """
    Construct a TensorDataset for skin cancer data.

    Args:
        data (List[Dict[str, Any]]): List of dictionaries containing image paths and labels.
        transform (Optional[Callable]): Transformation function to apply to the images. Defaults to None.
        num_workers (int): Number of workers for parallel processing. Defaults to 8.

    Returns:
        TensorDataset: A TensorDataset containing the processed images and labels.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda item: load_image(item, transform), data))

    data_list, targets_list = zip(*results)
    data_tensor = torch.stack(list(data_list))
    targets_tensor = torch.tensor(list(targets_list))

    return TensorDataset(data_tensor, targets_tensor)


def load_skin_cancer_data(
    data_dir: Path,
    dataset_name: str,
    batch_size: int,
    split_percents: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    sampler: Optional[LabelBasedSampler] = None,
    train_transform: Union[None, Callable] = None,
    val_transform: Union[None, Callable] = None,
    test_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Load skin cancer dataset (training, validation, and test set).

    Args:
        data_dir (Path): Directory containing the dataset files.
        dataset_name (str): Name of the dataset to load.
        batch_size (int): Batch size for the DataLoader.
        split_percents (Tuple[float, float, float]): Percentages for splitting the data into train, val, and test sets.
        sampler (Optional[LabelBasedSampler]): Sampler for the dataset. Defaults to None.
        train_transform (Union[None, Callable]): Transformations to apply to the training data. Defaults to None.
        val_transform (Union[None, Callable]): Transformations to apply to the validation data. Defaults to None.
        test_transform (Union[None, Callable]): Transformations to apply to the test data. Defaults to None.
        dataset_converter (Optional[DatasetConverter]): Converter to apply to the dataset. Defaults to None.
        seed (Optional[int]): Random seed for shuffling data. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]: DataLoaders for the training, validation,
            and test sets, and a dictionary with the number of examples in each set.
    """
    if sum(split_percents) != 1.0:
        raise ValueError("The split percentages must sum to 1.0")

    dataset_paths = {
        "Barcelona": data_dir.joinpath("ISIC_2019", "ISIC_19_Barcelona.json"),
        "Rosendahl": data_dir.joinpath("HAM10000", "HAM_rosendahl.json"),
        "Vienna": data_dir.joinpath("HAM10000", "HAM_vienna.json"),
        "UFES": data_dir.joinpath("PAD-UFES-20", "PAD_UFES_20.json"),
        "Canada": data_dir.joinpath("Derm7pt", "Derm7pt.json"),
    }

    if dataset_name not in dataset_paths:
        raise ValueError(f"Dataset {dataset_name} not found in available datasets.")

    dataset_path = dataset_paths[dataset_name]

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file {dataset_path} does not exist.\
            Please follow the instructions in fl4health/datasets/skin_cancer/README.md."
        )

    log(INFO, f"Data directory: {str(dataset_path)}")

    with open(dataset_path, "r") as f:
        data = json.load(f)["data"]

    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    total_size = len(data)
    train_size = int(split_percents[0] * total_size)
    valid_size = int(split_percents[1] * total_size)

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]
    test_data = data[train_size + valid_size :]

    # this is the default transform if more specific ones aren't defined.
    val_test_transform = transforms.Compose(
        [
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    )

    if train_transform is None:
        # this is the default transform if more specific ones aren't defined.
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
        val_transform = val_test_transform
    if test_transform is None:
        test_transform = val_test_transform

    train_ds: TensorDataset = construct_skin_cancer_tensor_dataset(train_data, transform=train_transform)
    valid_ds: TensorDataset = construct_skin_cancer_tensor_dataset(valid_data, transform=val_transform)
    test_ds: TensorDataset = construct_skin_cancer_tensor_dataset(test_data, transform=test_transform)

    if sampler is not None:
        train_ds = sampler.subsample(train_ds)
        valid_ds = sampler.subsample(valid_ds)
        test_ds = sampler.subsample(test_ds)

    if dataset_converter is not None:
        train_ds = dataset_converter.convert_dataset(train_ds)
        valid_ds = dataset_converter.convert_dataset(valid_ds)
        test_ds = dataset_converter.convert_dataset(test_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(valid_ds), "test_set": len(test_ds)}

    return train_loader, validation_loader, test_loader, num_examples
