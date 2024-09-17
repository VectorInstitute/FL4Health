import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self, transform: Optional[Callable], target_transform: Optional[Callable]) -> None:
        self.transform = transform
        self.target_transform = target_transform

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
        super().__init__(transform, target_transform)
        self.data = data
        self.targets = targets

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
    def __init__(
        self,
        data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        assert targets is not None, "SslTensorDataset targets must be None"

        super().__init__(data, targets, transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[index]

        assert self.target_transform is not None, "Target transform cannot be None."

        if self.transform is not None:
            data = self.transform(data)

        # Perform transform on input to yield target during data loading
        # More memory efficient than pre-computing transforms which requires
        # storing multiple copies of each sample
        transformed_data = self.target_transform(data)

        return data, transformed_data


class DictionaryDataset(Dataset):
    def __init__(self, data: Dict[str, List[torch.Tensor]], targets: torch.Tensor) -> None:
        """
        A torch dataset that supports a dictionary of input data rather than just a torch.Tensor. This kind of dataset
        is useful when dealing with non-trivial inputs to a model. For example, a language model may require token ids
        AND attention masks. This dataset supports that functionality.

        Args:
            data (Dict[str, List[torch.Tensor]]): A set of data for model training/input in the form of a dictionary
                of tensors.
            targets (torch.Tensor): Target tensor.
        """
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return {key: val[index] for key, val in self.data.items()}, self.targets[index]

    def __len__(self) -> int:
        first_key = list(self.data.keys())[0]
        return len(self.data[first_key])


class SyntheticDataset(TensorDataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        A dataset for synthetically created data strictly in the form of pytorch tensors. Generally, this dataset
        is just used for tests.
        Args:
            data (torch.Tensor): Data tensor with first dimension corresponding to the number of datapoints
            targets (torch.Tensor): Target tensor with first dimension corresponding to the number of datapoints
        """
        assert data.shape[0] == targets.shape[0]
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.targets is not None

        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self) -> int:
        return len(self.data)


D = TypeVar("D", bound=Union[TensorDataset, DictionaryDataset])


def select_by_indices(dataset: D, selected_indices: torch.Tensor) -> D:
    """
    This function is used to extract a subset of a dataset sliced by the indices in the tensor selected_indices. The
    dataset returned should be of the same type as the input but with only data associated with the given indices.

    Args:
        dataset (D): Dataset to be "subsampled" using the provided indices.
        selected_indices (torch.Tensor): Indices within the datasets data and targets (if they exist) to select

    Raises:
        TypeError: Will throw an error if the dataset provided is not supported

    Returns:
        D: Dataset with only the data associated with the provided indices. Must be of a supported type.
    """
    if isinstance(dataset, TensorDataset):
        modified_dataset = copy.deepcopy(dataset)
        modified_dataset.data = dataset.data[selected_indices]
        if dataset.targets is not None:
            modified_dataset.targets = dataset.targets[selected_indices]
        # cast being used here until the mypy bug mentioned in https://github.com/python/mypy/issues/12800 and the
        # duplicate ticket https://github.com/python/mypy/issues/10817 are fixed
        return cast(D, modified_dataset)
    elif isinstance(dataset, DictionaryDataset):
        new_targets = dataset.targets[selected_indices]
        new_data: Dict[str, List[torch.Tensor]] = {}
        for key, val in dataset.data.items():
            # Since val is a list of tensors, we can't directly index into it
            # using selected_indices.
            new_data[key] = [val[i] for i in selected_indices]
        # cast being used here until the mypy bug mentioned in https://github.com/python/mypy/issues/12800 and the
        # duplicate ticket https://github.com/python/mypy/issues/10817 are fixed
        return cast(D, DictionaryDataset(new_data, new_targets))
    else:
        raise TypeError("Dataset type is not supported by this function.")
