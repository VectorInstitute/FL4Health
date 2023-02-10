from logging import INFO
from pathlib import Path
from typing import Dict, Set, Tuple

import torch
import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_random_subsample(tensor_to_subsample: torch.Tensor, subsample_size: int) -> torch.Tensor:
    # NOTE: Assumes subsampling on rows
    tensor_size = tensor_to_subsample.shape[0]
    permutation = torch.randperm(tensor_size)
    return tensor_to_subsample[permutation[:subsample_size]]


def subsample_mnist_data(mnist_set: MNIST, minority_digits: Set[int], subsample_ratio: float) -> None:
    selected_indices_list = []
    for digit in range(10):
        indices_of_digit = (mnist_set.targets == digit).nonzero()
        if digit in minority_digits:
            subsample_size = int(indices_of_digit.shape[0] * subsample_ratio)
            subsampled_indices = get_random_subsample(indices_of_digit, subsample_size)
            selected_indices_list.append(subsampled_indices.squeeze())
        else:
            selected_indices_list.append(indices_of_digit.squeeze())
    selected_indices = torch.cat(selected_indices_list, dim=0)
    # Subsample labels and training data.
    mnist_set.targets = mnist_set.targets[selected_indices]
    mnist_set.data = mnist_set.data[selected_indices]


def load_data(
    data_dir: Path, batch_size: int, downsampling_ratio: float, minority_digits: Set[int]
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Load MNIST Dataset (training and validation set)."""
    log(INFO, f"Data directory: {str(data_dir)}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    training_set = MNIST(str(data_dir), train=True, download=True, transform=transform)
    validation_set = MNIST(str(data_dir), train=False, download=True, transform=transform)
    subsample_mnist_data(training_set, minority_digits, downsampling_ratio)
    subsample_mnist_data(validation_set, minority_digits, downsampling_ratio)

    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    num_examples = {
        "train_set": len(training_set),
        "validation_set": len(validation_set),
    }
    return train_loader, validation_loader, num_examples
