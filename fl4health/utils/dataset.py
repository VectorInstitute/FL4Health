from pathlib import Path
from typing import Callable, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.data: torch.Tensor
        self.targets: torch.Tensor
        self.transform: Union[Callable, None]
        self.target_transform: Union[Callable, None]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data.numpy())
        if self.target_transform is not None:
            target = self.target_transform(target.numpy())

        return data, target

    def __len__(self) -> int:
        return len(self.targets)


class MNISTDataset(BaseDataset):
    def __init__(
        self,
        data_path: Path,
        train: bool,
        transform: Union[None, Callable] = None,
        target_transform: Union[None, Callable] = None,
    ):
        mnist_dataset = MNIST(data_path, train=train, download=True)
        self.data = mnist_dataset.data
        self.targets = mnist_dataset.targets
        self.transform = transform
        self.target_transform = target_transform
