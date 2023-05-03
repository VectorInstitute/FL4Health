import math
from abc import ABC, abstractmethod
from typing import Any, List, Set, TypeVar

import numpy as np
import torch

from fl4health.utils.dataset import BaseDataset

T = TypeVar("T")


class LabelBasedSampler(ABC):
    """
    This is an abstract class to be extended to create dataset samplers
    based on the class of samples.
    """

    def __init__(self, unique_labels: List[Any]) -> None:
        self.unique_labels = unique_labels
        self.num_classes = len(self.unique_labels)

    @abstractmethod
    def subsample(self, dataset: BaseDataset) -> BaseDataset:
        raise NotImplementedError


class MinorityLabelBasedSampler(LabelBasedSampler):
    """
    This class is used to subsample a dataset so the classes are distributed in a non-IID way.
    In particular, the MinorityLabelBasedSampler explicitly downsamples classes based on the
    downsampling_ratio and minority_labels args used to construct the object. Subsampling a dataset is
    accomplished by calling the subsample method and passing a BaseDataset object. This will return
    the resulting subsampled dataset.
    """

    def __init__(self, unique_labels: List[T], downsampling_ratio: float, minority_labels: Set[T]) -> None:
        super().__init__(unique_labels)
        self.downsampling_ratio = downsampling_ratio
        self.minority_labels = minority_labels

    def subsample(self, dataset: BaseDataset) -> BaseDataset:
        """
        Returns a new dataset where samples part of minority_labels are downsampled
        """
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

        dataset.targets = dataset.targets[selected_indices]
        dataset.data = dataset.data[selected_indices]

        return dataset

    def _get_random_subsample(self, tensor_to_subsample: torch.Tensor, subsample_size: int) -> torch.Tensor:
        # NOTE: Assumes subsampling on rows
        tensor_size = tensor_to_subsample.shape[0]
        permutation = torch.randperm(tensor_size)
        return tensor_to_subsample[permutation[:subsample_size]]


class DirichletLabelBasedSampler(LabelBasedSampler):
    """
    class used to subsample a dataset so the classes of samples are distributed in a non-IID way.
    In particular, the DirichletLabelBasedSampler uses a dirichlet distribution to determine the number
    of samples from each class. The sampler is constructed by passing a beta parameter that determines
    the level of heterogeneity and a sample_percentage that determines the relative size of the modified
    dataset. Subsampling a dataset is accomplished by calling the subsample method and passing a BaseDataset object.
    This will return the resulting subsampled dataset.

    NOTE: The range for beta is (0, infinity). The larger the value of beta, the more evenly the multinomial
    probability of the labels will be. The smaller beta is the more heterogeneous it is.

    np.random.dirichlet([1]*5): array([0.23645891, 0.08857052, 0.29519184, 0.2999956 , 0.07978313])
    np.random.dirichlet([1000]*5): array([0.2066252 , 0.19644968, 0.20080513, 0.19992536, 0.19619462])
    """

    def __init__(self, unique_labels: List[Any], sample_percentage: float = 0.5, beta: float = 100) -> None:
        super().__init__(unique_labels)
        self.probabilities = np.random.dirichlet(np.repeat(beta, self.num_classes))
        self.sample_percentage = sample_percentage

    def subsample(self, dataset: BaseDataset) -> BaseDataset:
        """
        Returns a new dataset where samples are selected based on a dirichlet dsitribution over labels
        """
        total_num_samples = int(len(dataset) * self.sample_percentage)
        targets = dataset.targets

        class_idx_list = [torch.where(targets == target)[0].float() for target in self.unique_labels]

        num_samples_per_class = [math.ceil(prob * total_num_samples) for prob in self.probabilities]

        # For each class sample the given number of samples from the class specific indices
        # torch.multinomial is used to uniformly sample indices the size of given number of samples
        sampled_class_idx_list = [
            class_idx[torch.multinomial(torch.ones(class_idx.size(0)), num_samples, replacement=True)]
            for class_idx, num_samples in zip(class_idx_list, num_samples_per_class)
        ]

        selected_indices = torch.cat(sampled_class_idx_list, dim=0).long()

        # Due to precision errors with previous rounding, sum of sample counts
        # may differ from total_num_samples so we resample to ensure correct count
        selected_indices = selected_indices[:total_num_samples]

        dataset.targets = dataset.targets[selected_indices]
        dataset.data = dataset.data[selected_indices]

        return dataset
