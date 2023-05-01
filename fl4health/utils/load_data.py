from logging import INFO
from pathlib import Path
from typing import Dict, Tuple

import torchvision.transforms as transforms
from flwr.common.logger import log
from torch.utils.data import DataLoader

from fl4health.utils.dataset import BaseDataset, MNISTDataset
from fl4health.utils.sampler import LabelBasedSampler


def load_mnist_data(
    data_dir: Path,
    batch_size: int,
    sampler: LabelBasedSampler,
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
    train_ds = sampler.subsample(train_ds)
    val_ds = sampler.subsample(val_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=batch_size)

    num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
    return train_loader, validation_loader, num_examples
