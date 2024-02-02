from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10

from fl4health.utils.dataset import BaseDataset, MNISTDataset
from fl4health.utils.sampler import LabelBasedSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from opacus.data_loader import DPDataLoader

def load_mnist_data(
    data_dir: Path,
    batch_size: int,
    sampler: Optional[LabelBasedSampler] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load MNIST Dataset (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    train_ds: BaseDataset = MNISTDataset(data_dir, train=True, transform=transform)
    val_ds: BaseDataset = MNISTDataset(data_dir, train=False, transform=transform)

    if sampler is not None:
        train_ds = sampler.subsample(train_ds)
        val_ds = sampler.subsample(val_ds)

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
    evaluation_set = CIFAR10(str(data_dir), train=False, download=True, transform=transform)

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
    training_set = CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    validation_set = CIFAR10(str(data_dir), train=False, download=True, transform=transform)

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


def poisson_subsampler_cifar10(data_dir: Path, expected_batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """This dataloader for CIFAR-10 uses Poisson subsampling to select batches of given EXPECTED batch size.

    Args
        expected_batch_size: determines the Poisson sampling probability = expected_batch_size / total_data_size.
        For simplicity we will assume expected_batch_size is an integer. Note that this Poisson sampling probability will 
        differ between train & validation sets due to their total_data_size.
    """

    log(INFO, f"Data directory: {str(data_dir)}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    training_set = CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    validation_set = CIFAR10(str(data_dir), train=False, download=True, transform=transform)

    train_sampling_rate = expected_batch_size / len(training_set)
    validation_sampling_rate = expected_batch_size / len(validation_set)

    opcus_train_loader = DPDataLoader(training_set, sample_rate=train_sampling_rate)
    opcus_validation_loader = DPDataLoader(validation_set, sample_rate=validation_sampling_rate)


    # NOTE these are uniform samplers with fixed sample size 
    # poisson_sampler_training = RandomSampler(
    #     data_source=training_set, replacement=True, num_samples=number_of_samples, generator=torch.Generator(device='cuda')
    # )
    # poisson_sampler_validataion = RandomSampler(
    #     data_source=validation_set, replacement=True, num_samples=number_of_samples, generator=torch.Generator(device='cuda')
    # )
    # train_loader = DataLoader(training_set, batch_size=batch_size, sampler=poisson_sampler_training)
    # validation_loader = DataLoader(validation_set, batch_size=batch_size, sampler=poisson_sampler_validataion)


    train_size, _ = next(iter(opcus_train_loader))
    validation_size, _ = next(iter(opcus_validation_loader))

    num_examples = {
        "train_set": list(train_size.shape)[0],
        "validation_set": list(validation_size.shape)[0],
    }
    log(INFO, "---------------")
    log(INFO, num_examples)
    return opcus_train_loader, opcus_validation_loader, num_examples


