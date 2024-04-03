from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST


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

    def update_transform(self, f: Callable) -> None:
        if self.transform:
            original_transform = self.transform
            self.transform = lambda *x: f(original_transform(*x))
        else:
            self.transform = f

    def update_target_transform(self, g: Callable) -> None:
        if self.target_transform:
            original_target_transform = self.target_transform
            self.target_transform = lambda *x: g(original_target_transform(*x))
        else:
            self.target_transform = g


class MnistDataset(BaseDataset):
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


class Cifar10Dataset(BaseDataset):
    def __init__(
        self,
        data_path: Path,
        train: bool,
        transform: Union[None, Callable] = None,
        target_transform: Union[None, Callable] = None,
    ):
        cifar10_dataset = CIFAR10(data_path, train=train, download=True)
        self.data = torch.from_numpy(cifar10_dataset.data)
        self.targets = torch.Tensor(cifar10_dataset.targets).long()
        self.transform = transform
        self.target_transform = target_transform


class DictionaryDataset(Dataset):
    def __init__(self, data: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return {key: val[index] for key, val in self.data.items()}, self.targets[index]

    def __len__(self) -> int:
        first_key = list(self.data.keys())[0]
        return len(self.data[first_key])
