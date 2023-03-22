import math
from abc import ABC, abstractmethod
from typing import Any, List, Set

import numpy as np
import torch

from fl4health.utils.dataset import BaseDataset


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

    def __init__(self, unique_labels: List[Any], downsampling_ratio: float, minority_labels: Set[int]) -> None:
        super().__init__(unique_labels)
        self.downsampling_ratio = downsampling_ratio
        self.minority_labels = minority_labels

    def subsample(self, dataset: BaseDataset) -> BaseDataset:
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
    This class is used to subsample a dataset so the classes of samples are distributed in a non-IID way.
    In particular, the DirichletLabelBasedSampler uses a dirichlet distribution to determine the number
    of samples from each class. The sampler is constructed by passing a beta parameter that determines
    the level of heterogeneity and a sample_percentage that determines the relative size of the modified
    dataset. Subsampling a dataset is accomplished by calling the subsample method and passing a BaseDataset object.
    This will return the resulting subsampled dataset.
    """

    def __init__(self, unique_labels: List[Any], sample_percentage: float = 0.5, beta: float = 100) -> None:
        super().__init__(unique_labels)
        self.probabilities = np.random.dirichlet(np.repeat(beta, self.num_classes))
        self.sample_percentage = sample_percentage

    def subsample(self, dataset: BaseDataset) -> BaseDataset:
        total_num_samples = int(len(dataset) * self.sample_percentage)
        selected_indices_list: List[int] = []
        targets = dataset.targets.numpy()

        class_idx_list = [np.where(targets == target)[0] for target in self.unique_labels]
        for class_idx in class_idx_list:
            np.random.shuffle(class_idx)

        # Extend number of samples in classes when the sampled num_samples_per_class
        # exceeds actual number of samples in the class
        num_samples_per_class = [math.ceil(prob * total_num_samples) for prob in self.probabilities]
        mul_per_class = [
            math.ceil(num_samples / len(class_idx))
            for num_samples, class_idx in zip(num_samples_per_class, class_idx_list)
        ]

        class_idx_list = [class_idx.tolist() * mul for class_idx, mul in zip(class_idx_list, mul_per_class)]

        ss_class_idx_list = [
            class_idx[:num_samples] for num_samples, class_idx in zip(num_samples_per_class, class_idx_list)
        ]

        for class_idx in ss_class_idx_list:
            selected_indices_list.extend(class_idx)

        selected_indices = selected_indices_list[:total_num_samples]

        dataset.targets = dataset.targets[selected_indices]
        dataset.data = dataset.data[selected_indices]

        return dataset
