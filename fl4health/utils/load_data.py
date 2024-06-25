import random
from logging import INFO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

from fl4health.utils.dataset import TensorDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def split_data_and_targets(
    data: torch.Tensor, targets: torch.Tensor, validation_proportion: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    total_size = data.shape[0]
    train_size = int(total_size * (1 - validation_proportion))
    train_indices = random.sample(range(total_size), train_size)
    val_indices = [i for i in range(total_size) if i not in train_indices]
    train_data, train_targets = data[train_indices, ...], targets[train_indices, ...]
    val_data, val_targets = data[val_indices, ...], targets[val_indices, ...]
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
) -> Tuple[TensorDataset, TensorDataset]:
    data, targets = get_mnist_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(data, targets, validation_proportion)

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
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load MNIST Dataset (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
    training_set, validation_set = get_train_and_val_mnist_datasets(
        data_dir, transform, target_transform, validation_proportion
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


def get_cifar10_data_and_target_tensors(data_dir: Path, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    mnist_dataset = CIFAR10(data_dir, train=train, download=True)
    data = torch.Tensor(mnist_dataset.data)
    targets = torch.Tensor(mnist_dataset.targets).long()
    return data, targets


def get_train_and_val_cifar10_datasets(
    data_dir: Path,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    validation_proportion: float = 0.2,
) -> Tuple[TensorDataset, TensorDataset]:
    data, targets = get_cifar10_data_and_target_tensors(data_dir, True)

    train_data, train_targets, val_data, val_targets = split_data_and_targets(data, targets, validation_proportion)

    training_set = TensorDataset(train_data, train_targets, transform=transform, target_transform=target_transform)
    validation_set = TensorDataset(val_data, val_targets, transform=transform, target_transform=target_transform)

    return training_set, validation_set


def load_cifar10_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    validation_proportion: float = 0.2,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load CIFAR-10 (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set, validation_set = get_train_and_val_cifar10_datasets(data_dir, transform, None, validation_proportion)

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
    """Load CIFAR-10 test set only."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
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
