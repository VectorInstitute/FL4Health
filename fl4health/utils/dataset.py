from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self) -> None:
        self.transform: Union[Callable, None]
        self.target_transform: Union[Callable, None]

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

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.data: torch.Tensor = data
        self.targets: Optional[torch.Tensor] = targets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.targets is not None

        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)


class SslTensorDataset(TensorDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index]

        assert self.target_transform is not None, "Target transform cannot be None."

        if self.transform is not None:
            data = self.transform(data)

        # Perform transform on input to yield target during dataloading
        # More memory efficient than pre-computing transforms which requires
        # storing multiple copies of each sample
        transformed_data = self.target_transform(data)

        return data, transformed_data


class DictionaryDataset(Dataset):
    def __init__(self, data: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> None:
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return {key: val[index] for key, val in self.data.items()}, self.targets[index]

    def __len__(self) -> int:
        first_key = list(self.data.keys())[0]
        return len(self.data[first_key])
