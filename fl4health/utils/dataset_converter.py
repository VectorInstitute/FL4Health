from collections.abc import Callable
from functools import partial

import torch

from fl4health.utils.dataset import TensorDataset


class DatasetConverter(TensorDataset):
    def __init__(
        self,
        converter_function: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        dataset: TensorDataset | None,
    ) -> None:
        """
        Dataset converter classes are designed to re-format any dataset for a given training task, and to fit it
        into the unified training scheme of supervised learning in clients. Converters can be used in the data
        loading step. They can also apply a light pre-processing step on datasets before the training process.

        Args:
            converter_function (Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]): Function
                defining how the dataset should be converted.
            dataset (TensorDataset | None): Dataset to be converted.
        """
        assert dataset is None or dataset.targets is not None
        self.converter_function = converter_function
        self.dataset = dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Overriding this function from BaseDataset allows the converter to be compatible with the data transformers.
        ``converter_function`` is applied after the transformers.

        Args:
            index (int): The index of the batch of data to be extracted.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Get the raw batch data (input, target) and apply the
                ``converter_function`` before returning.
        """
        assert self.dataset is not None, "Error: no dataset is set, use convert_dataset(your_dataset: TensorDataset)"
        data, target = self.dataset.__getitem__(index)
        return self.converter_function(data, target)

    def __len__(self) -> int:
        """
        Returns the length of the dataset. Mostly just a wrapper on the standard pytorch dataset length to ensure that
        the dataset is not None.

        Returns:
            int: Dataset length.
        """
        assert self.dataset is not None, "Error: dataset is should be either converted or initiated."
        return len(self.dataset)

    def convert_dataset(self, dataset: TensorDataset) -> TensorDataset:
        """
        Applies the converter function over the dataset when the dataset is used
        (i.e. during the dataloader creation step).
        """
        # Dataset can be added/changed at any point in the pipeline.
        self.dataset = dataset
        # Returning this object as the converted dataset since this class overrides
        # the __getitem__ function of BaseDataset.
        return self


class AutoEncoderDatasetConverter(DatasetConverter):
    def __init__(
        self,
        condition: str | torch.Tensor | None = None,
        do_one_hot_encoding: bool = False,
        custom_converter_function: Callable | None = None,
        condition_vector_size: int | None = None,
    ) -> None:
        """
        A dataset converter specific to formatting supervised data such as MNIST for self-supervised training in
        autoencoder-based models, and potentially handling the existence of additional input (i.e. condition).

        This class includes three converter functions that are chosen based on the condition, other converter
        functions can be added or passed to support other conditions.

        Args:
            condition (str | torch.Tensor | None, optional): Could be a fixed tensor used for all the data samples,
                None for non-conditional models, or a name (str) passed for other custom conversions like "label".
                Defaults to None.
            do_one_hot_encoding (bool, optional): Should converter perform one hot encoding on the condition or not.
                Defaults to False.
            custom_converter_function (Callable | None, optional): User can define a new converter function. Defaults
                to None.
            condition_vector_size (int | None, optional): Size of the conditioning vector if available. Defaults to
                None.
        """
        self.condition = condition
        if isinstance(self.condition, torch.Tensor):
            # Condition should be a ready to use 1D tensor set in the client.
            assert self.condition.dim() == 1, (
                f"Error: condition should be a 1D vector instead of {self.condition.dim()}D tensor."
            )
        # Will be set in convert_dataset.
        self.data_shape: torch.Size
        self.do_one_hot_encoding = do_one_hot_encoding
        # If no converter function is passed, it should be defined here.
        if custom_converter_function is None:
            self.converter_function = self._setup_converter_function()
        else:
            self.converter_function = custom_converter_function
            assert condition_vector_size is not None, (
                "Error: The condition should be specified for a custom converter function."
            )
            self.condition_vector_size = condition_vector_size
        super().__init__(self.converter_function, dataset=None)

    def convert_dataset(self, dataset: TensorDataset) -> TensorDataset:
        assert dataset.targets is not None

        self.dataset = dataset
        # Data shape is saved to be used in the pack-unpack process.
        # This is the shape of the data after getting transformed by torch transforms.
        data, _ = self.dataset[0]
        self.data_shape = data.shape
        return self

    def get_condition_vector_size(self) -> int:
        if self.condition == "label":
            assert self.dataset is not None, "Error: no dataset is passed to the converter."
            assert self.dataset.targets is not None
            if self.do_one_hot_encoding:
                return len(torch.unique(self.dataset.targets))
            return len(self.dataset.targets[0])
        if isinstance(self.condition, torch.Tensor):
            return self.condition.size(0)
        if self.condition_vector_size is not None:
            return self.condition_vector_size
        if self.condition is None:
            return 0
        raise NotImplementedError("Error: support for this type of condition is not added to the data converter.")

    def _setup_converter_function(self) -> Callable:
        """
        Sets the converter function for autoencoder based models (if it is not already specified by the user).
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
        self, data: torch.Tensor, target: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The data converter function used for simple autoencoders or variational autoencoders."""
        # Target in self-supervised training for autoencoder is the data.
        return data, data

    def _cat_input_condition(
        self, data: torch.Tensor, target: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The data converter function used for conditional autoencoders.

        This converter is used when we have a torch tensor as condition for all the data samples.
        """
        # We can flatten the data since self.data_shape is already saved.
        # Target should be the original data.
        assert isinstance(self.condition, torch.Tensor), "Error: condition should be a torch tensor"
        return torch.cat([data.view(-1), self.condition]), data

    def _cat_input_label(self, data: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The data converter function used for conditional autoencoders.

        This converter is used when we want to condition each data sample on its label.
        """
        if self.do_one_hot_encoding:  # If condition needs to be one-hot encoded.
            # Create the condition vector by getting the one_hot encoded target.
            target = torch.nn.functional.one_hot(target, num_classes=self.get_condition_vector_size())
        # Concatenate the data and target (target is the condition)
        # We can flatten the data since self.data_shape is already saved.
        return torch.cat([data.view(-1), target]), data

    def get_unpacking_function(self) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        condition_vector_size = self.get_condition_vector_size()
        return partial(
            AutoEncoderDatasetConverter.unpack_input_condition,
            cond_vec_size=condition_vector_size,
            data_shape=self.data_shape,
        )

    @staticmethod
    def unpack_input_condition(
        packed_data: torch.Tensor, cond_vec_size: int, data_shape: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Unpacks model inputs (data and condition) from a tensor used in the training loop regardless of the
        converter function used to pack them. Unpacking relies on the size of the condition vector, and the original
        data shape which is saved before the packing process.

        Args:
            packed_data (torch.Tensor): Data tensor used in the training loop as the input to the model.
            cond_vec_size (int): Size of the conditional vector that has been packed into the ``packed_data`` variable.
            data_shape (torch.Size): Expected shape of the original data tensor after unpacking the conditioning
                vector.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Data in its original shape, and the condition vector to be fed into the
            model.
        """
        # We assume data is "batch first".
        x = packed_data[:, : -1 * cond_vec_size]  # Exclude the condition from input
        condition_matrix = packed_data[:, -1 * cond_vec_size :]  # Save the conditions
        # Data shape is restored in the unpacking process.
        assert data_shape is not None
        return x.view(-1, *data_shape), condition_matrix
