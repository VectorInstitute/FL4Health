import random
import warnings
from logging import INFO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.msd_dataset_sources import get_msd_dataset_enum, msd_md5_hashes, msd_urls
from fl4health.utils.sampler import LabelBasedSampler

with warnings.catch_warnings():
    # ignoring some annoying scipy deprecation warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from monai.apps.utils import download_and_extract


class ToNumpy:
    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.numpy()


def split_data_and_targets(
    data: torch.Tensor, targets: torch.Tensor, validation_proportion: float = 0.2, hash_key: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    total_size = data.shape[0]
    train_size = int(total_size * (1 - validation_proportion))
    if hash_key is not None:
        random.seed(hash_key)
    train_indices = random.sample(range(total_size), train_size)
    val_indices = [i for i in range(total_size) if i not in train_indices]
    train_data, train_targets = data[train_indices], targets[train_indices]
    val_data, val_targets = data[val_indices], targets[val_indices]
    return train_data, train_targets, val_data, val_targets


def get_mnist_data_and_target_tensors(data_dir: Path, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    mnist_dataset = MNIST(data_dir, train=train, download=True)
    data = torch.Tensor(mnist_dataset.data)
    targets = torch.Tensor(mnist_dataset.targets).long()
    return data, targets


def get_train_and_val_mnist_datasets(
    data_dir: Path,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    validation_proportion: float = 0.2,
    hash_key: Optional[int] = None,
) -> Tuple[TensorDataset, TensorDataset]:
    data, targets = get_mnist_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(
        data, targets, validation_proportion, hash_key
    )

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
    validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)
    return training_set, validation_set


def load_mnist_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
    validation_proportion: float = 0.2,
    hash_key: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Load MNIST Dataset (training and validation set).

    Args:
        data_dir (Path): The path to the MNIST dataset locally.
            Dataset is downloaded to this location if it does not already exist.
        batch_size (int): The batch size to use for the train and validation dataloader.
        sampler (Optional[LabelBasedSampler]): Optional sampler to subsample dataset based on labels.
        transform (Optional[Callable]): Optional transform to be applied to input samples.
        target_transform (Optional[Callable]): Optional transform to be applied to targets.
        dataset_converter (Optional[DatasetConverter]): Optional dataset converter used to convert
            the input and/or target of train and validation dataset.
        validation_proportion (float): A float between 0 and 1 specifying the proportion of samples
            to allocate to the validation dataset. Defaults to 0.2.
        hash_key (Optional[int]): Optional hash key to create a reproducible split for train and validation
            datasets.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: The train data loader, validation data loader
            and a dictionary with the sample counts of datasets underpinning the respective data loaders.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    if transform is None:
        transform = transforms.Compose(
            [
                ToNumpy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    training_set, validation_set = get_train_and_val_mnist_datasets(
        data_dir, transform, target_transform, validation_proportion, hash_key
    )

    if sampler is not None:
        training_set = sampler.subsample(training_set)
        validation_set = sampler.subsample(validation_set)

    if dataset_converter is not None:
        training_set = dataset_converter.convert_dataset(training_set)
        validation_set = dataset_converter.convert_dataset(validation_set)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    num_examples = {"train_set": len(training_set), "validation_set": len(validation_set)}
    return train_loader, validation_loader, num_examples


def load_mnist_test_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Load MNIST Test Dataset.

    Args:
        data_dir (Path): The path to the MNIST dataset locally.
            Dataset is downloaded to this location if it does not already exist.
        batch_size (int): The batch size to use for the test dataloader.
        sampler (Optional[LabelBasedSampler]): Optional sampler to subsample dataset based on labels.
        transform (Optional[Callable]): Optional transform to be applied to input samples.

    Returns:
        Tuple[DataLoader, Dict[str, int]]: The test data loader and a dictionary containing the sample count
            of the test dataset.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    if transform is None:
        transform = transforms.Compose(
            [
                ToNumpy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

    data, targets = get_mnist_data_and_target_tensors(data_dir, False)
    evaluation_set = TensorDataset(data, targets, transform)

    if sampler is not None:
        evaluation_set = sampler.subsample(evaluation_set)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples


def get_cifar10_data_and_target_tensors(data_dir: Path, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    cifar_dataset = CIFAR10(data_dir, train=train, download=True)
    data = torch.Tensor(cifar_dataset.data)
    targets = torch.Tensor(cifar_dataset.targets).long()
    return data, targets


def get_train_and_val_cifar10_datasets(
    data_dir: Path,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    validation_proportion: float = 0.2,
    hash_key: Optional[int] = None,
) -> Tuple[TensorDataset, TensorDataset]:
    data, targets = get_cifar10_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(
        data, targets, validation_proportion, hash_key
    )

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
    validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)

    return training_set, validation_set


def load_cifar10_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    validation_proportion: float = 0.2,
    hash_key: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Load CIFAR10 Dataset (training and validation set).

    Args:
        data_dir (Path): The path to the CIFAR10 dataset locally.
            Dataset is downloaded to this location if it does not already exist.
        batch_size (int): The batch size to use for the train and validation dataloader.
        sampler (Optional[LabelBasedSampler]): Optional sampler to subsample dataset based on labels.
        validation_proportion (float): A float between 0 and 1 specifying the proportion of samples
            to allocate to the validation dataset. Defaults to 0.2.
        hash_key (Optional[int]): Optional hash key to create a reproducible split for train and validation
            datasets.

    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]: The train data loader, validation data loader
            and a dictionary with the sample counts of datasets underpinning the respective data loaders.
    """
    log(INFO, f"Data directory: {str(data_dir)}")

    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set, validation_set = get_train_and_val_cifar10_datasets(
        data_dir, transform, None, validation_proportion, hash_key
    )

    if sampler is not None:
        training_set = sampler.subsample(training_set)
        validation_set = sampler.subsample(validation_set)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }
    return train_loader, validation_loader, num_examples


def load_cifar10_test_data(
    data_dir: Path, batch_size: int, sampler: Optional[LabelBasedSampler] = None
) -> Tuple[DataLoader, Dict[str, int]]:
    """
    Load CIFAR10 Test Dataset.

    Args:
        data_dir (Path): The path to the CIFAR10 dataset locally.
            Dataset is downloaded to this location if it does not already exist.
        batch_size (int): The batch size to use for the test dataloader.
        sampler (Optional[LabelBasedSampler]): Optional sampler to subsample dataset based on labels.

    Returns:
        Tuple[DataLoader, Dict[str, int]]: The test data loader and a dictionary containing the sample count
            of the test dataset.
    """
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            ToNumpy(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    data, targets = get_cifar10_data_and_target_tensors(data_dir, False)
    evaluation_set = TensorDataset(data, targets, transform)

    if sampler is not None:
        evaluation_set = sampler.subsample(evaluation_set)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples


def load_msd_dataset(data_path: str, msd_dataset_name: str) -> None:
    """
    Downloads and extracts one of the 10 Medical Segmentation Decathelon (MSD)
    datasets.

    Args:
        data_path (str): Path to the folder in which to extract the
            dataset. The data itself will be in a subfolder named after the
            dataset, not in the data_path directory itself. The name of the
            folder will be the name of the dataset as defined by the values of
            the MsdDataset enum returned by get_msd_dataset_enum
        msd_dataset_name (str): One of the 10 msd datasets
    """
    msd_enum = get_msd_dataset_enum(msd_dataset_name)
    msd_hash = msd_md5_hashes[msd_enum]
    url = msd_urls[msd_enum]
    download_and_extract(url=url, output_dir=data_path, hash_val=msd_hash, hash_type="md5", progress=True)
