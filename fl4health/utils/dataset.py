import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar, cast

import torch
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    def __init__(self, transform: Callable | None, target_transform: Callable | None) -> None:
        """
        Abstract base class for datasets used in this library. This class inherits from the torch Dataset base class.

        Args:
            transform (Callable | None, optional): Optional transformation to be applied to the input data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
            target_transform (Callable | None, optional): Optional transformation to be applied to the target data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
        """
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
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to be implemented by any inheriting dataset to produce a data value at provided index from
        the underlying data.

        Args:
            index (int): Index at which to extract the data from the dataset.

        Raises:
            NotImplementedError: Throws if one attempts to use this function.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Input and target tensors extracted at the provided index.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to be implemented by any inheriting dataset to produce a length value for the underlying data.

        Raises:
            NotImplementedError: Throws if one attempts to use this function.

        Returns:
            (int): Length of the underlying data.
        """
        raise NotImplementedError


class TensorDataset(BaseDataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Basic dataset where the data and targets are assumed to be torch tensors. Optionally, this class allows the
        user to perform transformations on both the data and the targets. This is useful, for example, in performing
        data augmentation, label blurring, etc.

        Args:
            data (torch.Tensor): Input data for training.
            targets (torch.Tensor | None, optional): Target data for training. Defaults to None.
            transform (Callable | None, optional): Optional transformation to be applied to the input data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
            target_transform (Callable | None, optional): Optional transformation to be applied to the target data.

                **NOTE**: This transformation is applied at load time within ``__get_item__``

                Defaults to None.
        """
        super().__init__(transform, target_transform)
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the data and targets from the dataset at the provided index. Transformations are performed, as
        specified in this datasets ``transform`` and ``target_transform`` functions. These are independently
        applied to the data and targets, respectively.

        Args:
            index (int): Index in the dataset to extract.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Input data at the index after applying ``transform`` if any, targets
                after applying ``target_transform`` if any.
        """
        assert self.targets is not None

        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        """
        Length of the dataset as determined by len() applied to torch dataset.

        Returns:
            (int): Length of dataset.
        """
        return len(self.data)


class SslTensorDataset(TensorDataset):
    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Dataset specifically designed to perform self-supervised learning, where we don't have a specific set of
        targets, because targets are derived from the data tensors.

        Args:
            data (torch.Tensor): Tensor representing the input data for the dataset.
            targets (torch.Tensor | None, optional): **REQUIRED TO BE NONE**. The type and argument here is simply to
                maintain compatibility with our ``TensorDataset`` base. Defaults to None.
            transform (Callable | None, optional): Any transform to be applied to the data tensors. This transform is
                performed BEFORE and target transforms that produce the self-supervised targets from the data.

                **NOTE**: These transformations and the ``target_transform`` functions are applied **AT LOAD TIME**.

                Defaults to None.
            target_transform (Callable | None, optional): Any transform to be applied to the data tensors to produce
                target tensors for training. This transform is performed after and transforms for the data tensors
                themselves to produce the self-supervised targets from the data.

                **NOTE**: These transformation functions are applied **AT LOAD TIME**.

                Defaults to None.
        """
        assert targets is None, "SslTensorDataset targets must be None"

        super().__init__(data, targets, transform, target_transform)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the data and targets from the dataset at the provided index. Because this is an self-supervised
        learning dataset. The input data also serves as the target (subject to transformations). Transformations
        are performed, as specified in this datasets ``transform`` and ``target_transform`` functions. These are
        applied first to the data, then targets are created by applying ``target_transform`` to the result.

        Args:
            index (int): Index in the dataset to extract.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Input data at the index after applying ``transform`` if any, targets
                derived from data after applying ``transform`` and ``target_transform`` in sequence.
        """
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
    def __init__(self, data: dict[str, list[torch.Tensor]], targets: torch.Tensor) -> None:
        """
        A torch dataset that supports a dictionary of input data rather than just a ``torch.Tensor``. This kind of
        dataset is useful when dealing with non-trivial inputs to a model. For example, a language model may require
        token ids AND attention masks. This dataset supports that functionality.

        Args:
            data (dict[str, list[torch.Tensor]]): A set of data for model training/input in the form of a dictionary
                of tensors.
            targets (torch.Tensor): Target tensor.
        """
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Extracts data from the ``DictionaryDataset`` at the provided index. The targets are simply extracted directly
        using index. The input data dictionary is iterated through and each piece of data in the dictionary values
        is extracted at the provided index and "re-wrapped" as a dictionary.

        Args:
            index (int): Index of the data to be extracted from the dataset.

        Returns:
            (tuple[dict[str, torch.Tensor], torch.Tensor]): Dictionary with the same keys as the dataset data
                dictionary with data extracted at the provided index, target data extracted from the targets tensor at
                index.
        """
        return {key: val[index] for key, val in self.data.items()}, self.targets[index]

    def __len__(self) -> int:
        """
        Gets the length of the dataset as extracted from the first piece of data in the data dictionary.

        **NOTE**: This implicitly assumes that the length of the data in each entry of the dictionary of data is
        uniform.

        Returns:
            (int): Dataset length.
        """
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the data at the provided index.

        Args:
            index (int): Index of the data in the dataset to be returned

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Input and targets at the provided index.
        """
        assert self.targets is not None

        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self) -> int:
        """
        Returns the length of the dataset. Identical to the pytorch dataset length function.

        Returns:
            (int): Length of the data.
        """
        return len(self.data)


D = TypeVar("D", bound=TensorDataset | DictionaryDataset)


def select_by_indices(dataset: D, selected_indices: torch.Tensor) -> D:
    """
    This function is used to extract a subset of a dataset sliced by the indices in the tensor ``selected_indices``.
    The dataset returned should be of the same type as the input but with only data associated with the given indices.

    Args:
        dataset (D): Dataset to be "subsampled" using the provided indices.
        selected_indices (torch.Tensor): Indices within the datasets data and targets (if they exist) to select.

    Raises:
        TypeError: Will throw an error if the dataset provided is not supported.

    Returns:
        (D): Dataset with only the data associated with the provided indices. Must be of a supported type.
    """
    if isinstance(dataset, TensorDataset):
        modified_dataset = copy.deepcopy(dataset)
        modified_dataset.data = dataset.data[selected_indices]
        if dataset.targets is not None:
            modified_dataset.targets = dataset.targets[selected_indices]
        # cast being used here until the mypy bug mentioned in https://github.com/python/mypy/issues/12800 and the
        # duplicate ticket https://github.com/python/mypy/issues/10817 are fixed
        return cast(D, modified_dataset)
    if isinstance(dataset, DictionaryDataset):
        new_targets = dataset.targets[selected_indices]
        new_data: dict[str, list[torch.Tensor]] = {}
        for key, val in dataset.data.items():
            # Since val is a list of tensors, we can't directly index into it
            # using selected_indices.
            new_data[key] = [val[i] for i in selected_indices]
        # cast being used here until the mypy bug mentioned in https://github.com/python/mypy/issues/12800 and the
        # duplicate ticket https://github.com/python/mypy/issues/10817 are fixed
        return cast(D, DictionaryDataset(new_data, new_targets))
    raise TypeError("Dataset type is not supported by this function.")
