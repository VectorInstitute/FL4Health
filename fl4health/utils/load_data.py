from logging import INFO
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader

from fl4health.utils.dataset import BaseDataset, Cifar10Dataset, MnistDataset
from fl4health.utils.dataset_converter import DatasetConverter
from fl4health.utils.sampler import LabelBasedSampler


def load_mnist_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
    transform: Union[None, Callable] = None,
    target_transform: Union[None, Callable] = None,
    dataset_converter: Optional[DatasetConverter] = None,
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
    train_ds: BaseDataset = MnistDataset(data_dir, train=True, transform=transform)
    val_ds: BaseDataset = MnistDataset(data_dir, train=False, transform=transform)

    if sampler is not None:
        train_ds = sampler.subsample(train_ds)
        val_ds = sampler.subsample(val_ds)

    if dataset_converter is not None:
        train_ds = dataset_converter.convert_dataset(train_ds)
        val_ds = dataset_converter.convert_dataset(val_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
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
    evaluation_set: BaseDataset = Cifar10Dataset(data_dir, train=False, transform=transform)

    if sampler is not None:
        evaluation_set = sampler.subsample(evaluation_set)

    evaluation_loader = DataLoader(evaluation_set, batch_size=batch_size, shuffle=False)
    num_examples = {"eval_set": len(evaluation_set)}
    return evaluation_loader, num_examples


def load_cifar10_data(
    data_dir: Path, batch_size: int, sampler: Optional[LabelBasedSampler] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load CIFAR-10 (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set: BaseDataset = Cifar10Dataset(data_dir, train=True, transform=transform)
    validation_set: BaseDataset = Cifar10Dataset(data_dir, train=False, transform=transform)

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
