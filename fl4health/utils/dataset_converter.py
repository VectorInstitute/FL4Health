from abc import ABC
from typing import Callable, Optional, Tuple, Union

import torch

from fl4health.utils.dataset import BaseDataset


class DatasetConverter(ABC, BaseDataset):
    """Dataset converter classes are designed to re-format any dataset for a given training task,
    and to fit it into the unified training scheme of supervised learning in clients.
    Converters can be used in the data loading step. They can also apply a light
    pre-processing step on datasets before the training process.
    """

    def __init__(self, converter_function: Callable, dataset: Union[None, BaseDataset]) -> None:
        self.converter_function = converter_function
        self.dataset = dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Overriding this function from BaseDataset allows the converter to be compatible with the data transformers.
        # converter_function is applied after the transformers.
        assert self.dataset is not None, "Error: no dataset is set, use convert_dataset(your_dataset: BaseDataset)"
        data, target = self.dataset.__getitem__(index)
        return self.converter_function(data, target)

    def __len__(self) -> int:
        assert self.dataset is not None, "Error: dataset is should be either converted or initiated."
        return len(self.dataset)

    def convert_dataset(self, dataset: BaseDataset) -> BaseDataset:
        """Applies the converter function over the dataset when the dataset is used
        (i.e. during the dataloader creation step)."""
        # Dataset can be added/changed at any point in the pipeline.
        self.dataset = dataset
        # Returning this object as the converted dataset since this class overrides
        # the __getitem__ function of BaseDataset.
        return self


class AutoEncoderDatasetConverter(DatasetConverter):
    """A dataset converter specific to formating supervised data such as MNIST for
    self-supervised training in autoencoder-based models, and potentially handling
    the existence of additional input i.e. condition.
    This class includes three converter functions that are chosen based on the condition,
    other converter functions can be added or passed to support other conditions.
    """

    def __init__(
        self,
        condition: Union[None, str, torch.Tensor],
        custom_converter_function: Optional[Callable] = None,
    ):
        self.condition = condition
        if isinstance(self.condition, torch.Tensor):
            # Condition should be a ready to use 1D tensor set in the client.
            assert (
                self.condition.dim() == 1
            ), f"Error: condition should be a 1D vector instead of {self.condition.dim()}D tensor."
        # self.data_shape is initialized with zero as we still haven't converted any dataset.
        # Will be set in convert_dataset.
        self.data_shape = torch.Size([0])
        # If no converter function is passed, it should be defined here.
        if custom_converter_function is None:
            self.converter_function = self._setup_converter_function()
        else:
            self.converter_function = custom_converter_function
        super().__init__(self.converter_function, dataset=None)

    def convert_dataset(self, dataset: BaseDataset) -> BaseDataset:
        self.dataset = dataset
        # if self.data_shape is None:
        #     # Data shape is saved to be used in the pack-unpack process.
        #     # This is the shape of the data after getting treansformed by torch transforms.
        data, _ = self.dataset[0]
        self.data_shape = data.shape
        return self

    def get_condition_vector_size(self) -> int:
        if self.condition is None:
            return 0
        elif self.condition == "label":
            assert self.dataset is not None, "Error: no dataset is passed to the converter."
            return len(torch.unique(self.dataset.targets))
        elif isinstance(self.condition, torch.Tensor):
            return self.condition.size(0)
        else:
            raise NotImplementedError("Error: support for this type of condition is not added to the data converter.")

    def _setup_converter_function(self) -> Callable:
        """Sets the converter function for autoencoder based models (if it is not already specified by the user).
        If condition is not None, the respective converter function accounts for the type of
        condition and handles the concatenation.

        Returns:
            Callable: The suitable converter function based on the condition.
        """
        # If the autoencoder is conditional
        if self.condition is not None:
            if self.condition == "label":
                # Condition depends on the target.
                converter_function = self._cat_input_label
            elif isinstance(self.condition, torch.Tensor):
                converter_function = self._cat_input_condition
            else:
                raise NotImplementedError("Error: support for this type of condition is not added.")

        elif self.condition is None:  # non-conditional autoencoder
            converter_function = self._only_replace_target_with_data
        return converter_function

    def _only_replace_target_with_data(
        self, data: torch.Tensor, target: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The data converter function used for simple autoencoders or variational autoencoders."""
        # Target in self-supervised training for autoencoder is the data.
        return data, data

    def _cat_input_condition(
        self, data: torch.Tensor, target: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The data converter function used for conditional autoencoders.
        This converter is used when we have a torch tensor as condition for all the data samples.
        """
        # We can flatten the data since self.data_shape is already saved.
        # Target should be the original data.
        assert isinstance(self.condition, torch.Tensor), "Error: condition should be a torch tensor"
        return torch.cat([data.view(-1), self.condition]), data

    def _cat_input_label(self, data: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The data converter function used for conditional autoencoders.
        This converter is used when we want to condition each data sample on its label.
        """
        num_conditions = self.get_condition_vector_size()
        # Create the condition vector by getting the one_hot encoded target.
        one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_conditions)
        # Concatenate the data and target (target is the condition)
        # We can flatten the data since self.data_shape is already saved.
        # Target should be the original data.
        return torch.cat([data.view(-1), one_hot_target]), data

    def unpack_input_condition(self, packed_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpacks model inputs (data and condition) from a tensor used in the training loop
        regardless of the converter function used to pack them. Unpacking relies on the size of the condition vector,
        and the original data shape which is saved before the packing process.

        Args:
            packed_data (torch.Tensor): Data tensor used in the training loop as the input to the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data in its original shape, and the condition vector
            to be fed into the model.
        """
        # We assume data is "batch first".
        x = packed_data[:, : -1 * self.get_condition_vector_size()]  # Exclude the condition from input
        condition_matrix = packed_data[:, -1 * self.get_condition_vector_size() :]  # Save the conditions
        # Data shape is restored in the unpacking process.
        assert self.data_shape is not None
        return x.view(-1, *self.data_shape), condition_matrix
