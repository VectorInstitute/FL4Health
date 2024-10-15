import math
from abc import ABC, abstractmethod
from logging import INFO, WARN
from typing import Any, List, Optional, Set, TypeVar, Union

import numpy as np
import torch
from flwr.common.logger import log

from fl4health.utils.dataset import DictionaryDataset, TensorDataset, select_by_indices

T = TypeVar("T")
D = TypeVar("D", bound=Union[TensorDataset, DictionaryDataset])


class LabelBasedSampler(ABC):

    def __init__(self, unique_labels: List[Any]) -> None:
        """
        This is an abstract class to be extended to create dataset samplers based on the class of samples.

        Args:
            unique_labels (List[Any]): The full set of labels contained in the dataset.
        """
        self.unique_labels = unique_labels
        self.num_classes = len(self.unique_labels)

    @abstractmethod
    def subsample(self, dataset: D) -> D:
        raise NotImplementedError


class MinorityLabelBasedSampler(LabelBasedSampler):
    def __init__(self, unique_labels: List[T], downsampling_ratio: float, minority_labels: Set[T]) -> None:
        """
        This class is used to subsample a dataset so the classes are distributed in a non-IID way.
        In particular, the MinorityLabelBasedSampler explicitly downsamples classes based on the
        downsampling_ratio and minority_labels args used to construct the object. Subsampling a dataset is
        accomplished by calling the subsample method and passing a BaseDataset object. This will return
        the resulting subsampled dataset.

        Args:
            unique_labels (List[T]): The full set of labels contained in the dataset.
            downsampling_ratio (float): The percentage to which the specified "minority" labels are downsampled. For
                example, if a label L has 10 examples and the downsampling_ratio is 0.2, then 8 of the datapoints with
                label L are discarded.
            minority_labels (Set[T]): The labels subject to downsampling.
        """
        super().__init__(unique_labels)
        self.downsampling_ratio = downsampling_ratio
        self.minority_labels = minority_labels

    def subsample(self, dataset: D) -> D:
        """
        Returns a new dataset where samples part of minority_labels are downsampled

        Args:
            dataset (D): Dataset to be modified, through downsampling on specified labels.

        Returns:
            D: New dataset with downsampled labels.
        """
        assert dataset.targets is not None, "A label-based sampler requires targets but this dataset has no targets"
        selected_indices_list: List[torch.Tensor] = []
        for label in self.unique_labels:
            # Get indices of samples equal to the current label
            indices_of_label = (dataset.targets == label).nonzero()
            if label in self.minority_labels:
                subsample_size = int(indices_of_label.shape[0] * self.downsampling_ratio)
                subsampled_indices = self._get_random_subsample(indices_of_label, subsample_size)
                selected_indices_list.append(subsampled_indices.squeeze())
            else:
                selected_indices_list.append(indices_of_label.squeeze())

        selected_indices = torch.cat(selected_indices_list, dim=0)

        return select_by_indices(dataset, selected_indices)

    def _get_random_subsample(self, tensor_to_subsample: torch.Tensor, subsample_size: int) -> torch.Tensor:
        """
        Given a tensor a new tensor is created by selecting a set of rows from the original tensor of
        size subsample_size

        Args:
            tensor_to_subsample (torch.Tensor): Tensor to be subsampled. Assumes that we're subsampling rows of the
                tensor
            subsample_size (int): How many rows we want to extract from the tensor.

        Returns:
            torch.Tensor: New tensor with subsampled rows
        """
        # NOTE: Assumes subsampling on rows
        tensor_size = tensor_to_subsample.shape[0]
        assert subsample_size < tensor_size
        permutation = torch.randperm(tensor_size)
        return tensor_to_subsample[permutation[:subsample_size]]


class DirichletLabelBasedSampler(LabelBasedSampler):
    def __init__(
        self,
        unique_labels: List[Any],
        hash_key: Optional[int] = None,
        sample_percentage: float = 0.5,
        beta: float = 100,
    ) -> None:
        """
        class used to subsample a dataset so the classes of samples are distributed in a non-IID way.
        In particular, the DirichletLabelBasedSampler uses a dirichlet distribution to determine the number
        of samples from each class. The sampler is constructed by passing a beta parameter that determines
        the level of heterogeneity and a sample_percentage that determines the relative size of the modified
        dataset. Subsampling a dataset is accomplished by calling the subsample method and passing a BaseDataset
        object. This will return the resulting subsampled dataset.

        NOTE: The range for beta is (0, infinity). The larger the value of beta, the more evenly the multinomial
        probability of the labels will be. The smaller beta is the more heterogeneous it is.

        np.random.dirichlet([1]*5): array([0.23645891, 0.08857052, 0.29519184, 0.2999956 , 0.07978313])
        np.random.dirichlet([1000]*5): array([0.2066252 , 0.19644968, 0.20080513, 0.19992536, 0.19619462])

        Args:
            unique_labels (List[Any]): The full set of labels contained in the dataset.
            sample_percentage (float, optional): The downsampling of the entire dataset to do. For example, if this
                value is 0.5 and the dataset is of size 100, we will end up with 50 total data points. Defaults to 0.5.
            beta (float, optional): This controls the heterogeneity of the label sampling. The smaller the beta, the
                more skewed the label assignments will be for the dataset. Defaults to 100.
            hash_key (Optional[int], optional): Seed for the random number generators and samplers. Defaults to None.
        """
        super().__init__(unique_labels)

        self.hash_key = hash_key

        self.torch_generator = None
        if self.hash_key is not None:
            log(INFO, f"Setting seed to {self.hash_key} for the Torch and Numpy Generators")
            log(WARN, "Note that setting a hash key here will override any torch and numpy seeds that you have set")
            self.torch_generator = torch.Generator().manual_seed(self.hash_key)
            np_generator = np.random.default_rng(self.hash_key)
            self.probabilities = np_generator.dirichlet(np.repeat(beta, self.num_classes))
        else:
            self.probabilities = np.random.dirichlet(np.repeat(beta, self.num_classes))
        log(INFO, f"Setting probabilities to {self.probabilities}")

        self.sample_percentage = sample_percentage

    def subsample(self, dataset: D) -> D:
        """
        Returns a new dataset where samples are selected based on a dirichlet distribution over labels

        Args:
            dataset (D): Dataset to be modified, through downsampling on specified labels.

        Returns:
            D: New dataset with downsampled labels.
        """
        assert dataset.targets is not None, "A label-based sampler requires targets but this dataset has no targets"
        assert self.sample_percentage <= 1.0

        total_num_samples = int(len(dataset) * self.sample_percentage)
        targets = dataset.targets

        class_idx_list = [torch.where(targets == target)[0].float() for target in self.unique_labels]

        num_samples_per_class = [math.ceil(prob * total_num_samples) for prob in self.probabilities]

        # For each class sample the given number of samples from the class specific indices
        # torch.multinomial is used to uniformly sample indices the size of given number of samples
        sampled_class_idx_list = [
            class_idx[
                torch.multinomial(
                    torch.ones(class_idx.size(0)), num_samples, replacement=True, generator=self.torch_generator
                )
            ]
            for class_idx, num_samples in zip(class_idx_list, num_samples_per_class)
        ]
        selected_indices = torch.cat(sampled_class_idx_list, dim=0).long()

        # Due to precision errors with previous rounding, sum of sample counts
        # may differ from total_num_samples so we resample to ensure correct count
        selected_indices = selected_indices[:total_num_samples]

        return select_by_indices(dataset, selected_indices)
