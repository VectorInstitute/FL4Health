import math
from logging import INFO, WARN
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from flwr.common.logger import log

from fl4health.utils.dataset import DictionaryDataset, TensorDataset, select_by_indices

T = TypeVar("T")
D = TypeVar("D", bound=Union[TensorDataset, DictionaryDataset])


class DirichletLabelBasedAllocation(Generic[T]):
    def __init__(
        self,
        number_of_partitions: int,
        unique_labels: List[T],
        min_label_examples: Optional[int] = None,
        beta: Optional[float] = None,
        prior_distribution: Optional[Dict[T, np.ndarray]] = None,
    ) -> None:
        """
        The class supports partitioning of a dataset into a set of datasets (of the same type) via Dirichlet
        allocation. That is, for each label, a Dirichlet distribution is constructed using beta across a requested
        number of partitions. Data associated with the label are then assigned to each partition according to the
        distribution. Another distribution is sampled for the next label, and so on.

        NOTE: This differs in kind from label-based Dirichlet sampling. There, an existing dataset is subsampled in
        place (rather than partitioned) such that its labels match a Dirichlet distribution.

        NOTE: The range for beta is (0, infinity). The larger the value of beta, the more uniform the multinomial
        probability of the clients will be. The smaller beta is the more heterogeneous it is.
        np.random.dirichlet([1]*5): array([0.23645891, 0.08857052, 0.29519184, 0.2999956 , 0.07978313])
        np.random.dirichlet([1000]*5): array([0.2066252 , 0.19644968, 0.20080513, 0.19992536, 0.19619462])

        Example Usage:
            original_dataset = SyntheticDataset(
                torch.rand((10000, 3, 3)),
                torch.randint(low=0, high=10, size=(10000, 1))
            )
            heterogeneous_partitioner = DirichletLabelBasedAllocation(
                number_of_partitions=10, unique_labels=list(range(10)), beta=10.0, min_label_examples=2
            )
            partitioned_datasets = heterogeneous_partitioner.partition_dataset(original_dataset, max_retries=5)

        Args:
            number_of_partitions (int): Number of new datasets that we want to break the current dataset into
            unique_labels (List[T]): This is the set of labels through which we'll iterate to perform allocation
            min_label_examples (Optional[int], optional): This is an optional input if you want to ensure a minimum
                number of labels is present on each partition. If prior distribution is provided, this is ignored.
                NOTE: This does not guarantee feasibility. That is, if you have a very small beta and request a large
                minimum number here, you are unlikely to satisfy this request. In partitioning, if the minimum isn't
                satisfied, we resample from the Dirichlet distribution. This is repeated some limited number of times.
                Otherwise the partitioner "gives up". Defaults to None.
            beta (Optional[float]): This controls the heterogeneity of the partition allocations. The smaller the beta,
              the more skewed the label assignments will be to different clients. It is mutually exclusive with given
              prior distribution.
            prior_distribution (Optional[Dict[T, np.ndarray]], optional): This is an optional input if you want to
              provide a prior distribution for the Dirichlet distribution. This is useful if you want to make sure that
              the partitioning of test data is similar to the partitioning of the training data. Defaults to None. It
              is mutually exclusive with the beta parameter and min_label_examples.
        """
        assert (beta is not None) ^ (
            prior_distribution is not None
        ), "Either beta or a prior distribution must be provided, but not both."
        self.number_of_partitions = number_of_partitions
        self.unique_labels = unique_labels
        self.n_unique_labels = len(unique_labels)
        self.beta = beta
        self.min_label_examples = min_label_examples if min_label_examples else 0
        self.prior_distribution = prior_distribution
        if self.prior_distribution is not None:
            assert (
                len(self.prior_distribution) == self.n_unique_labels
            ), "The length of the prior must match the number of labels"
            if self.min_label_examples > 0:
                log(
                    WARN,
                    "A prior distribution has been provided for the partitioner",
                    "so min_label_examples will be ignored.",
                )

    def partition_label_indices(
        self, label: T, label_indices: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int, np.ndarray]:
        """
        Given a set of indices from the dataset corresponding to a particular label, the indices are allocated using
        a Dirichlet distribution, to the partitions.

        Args:
            label (T): Label is passed for logging transparency. It must be convertible to a string through str()
            label_indices (torch.Tensor): Indices from the dataset corresponding to a particular label. This assumes
                that the tensor is 1D and it's len constitutes the number of total datapoints with the label.

        Returns:
            List[torch.Tensor]: partitioned indices of datapoints with the corresponding label.
            int: The minimum number of data points assigned to a partition.
            np.ndarray: The Dirichlet distribution used to partition the data points.
        """
        if self.prior_distribution is not None:
            label_prior_distribution = self.prior_distribution[label]
            assert (
                len(label_prior_distribution) == self.number_of_partitions
            ), f"The length of the prior distribution for label ({str(label)}) must match the number of partitions"
            if sum(label_prior_distribution) != 1:
                log(
                    WARN,
                    f"The provided prior distribution for label ({str(label)}) does not sum to 1. "
                    "It will be normalized to sum to 1.",
                )
            partition_allocations = label_prior_distribution / sum(label_prior_distribution)
            log(
                INFO,
                f"The allocation distribution for label ({str(label)}) is {partition_allocations} "
                "using the provided prior distribution",
            )
        elif self.beta is not None:
            # These are the percentages of the label indices to be distributed for each partition
            partition_allocations = np.random.dirichlet(np.repeat(self.beta, self.number_of_partitions))
            log(
                INFO,
                (
                    f"The allocation distribution for label ({str(label)}) is {partition_allocations} "
                    f"using a beta of {self.beta}",
                ),
            )
        else:
            raise ValueError("Either beta or a prior distribution must be provided.")
        total_data_points_with_label = len(label_indices)
        num_samples_per_partition = [
            math.floor(probability * total_data_points_with_label) for probability in partition_allocations
        ]
        log(INFO, f"Assignment of datapoints to partitions as {num_samples_per_partition}")
        # Getting the smallest sample across the partitions to decided if it was acceptable.
        min_samples = min(num_samples_per_partition)
        total_samples_in_partitions = sum(num_samples_per_partition)
        # Due to rounding, we need to make sure the partitions we're asking for sum to the total data points size.
        # So our final partition will "fill" the difference and be discarded
        num_samples_per_partition.append(total_data_points_with_label - total_samples_in_partitions)

        shuffled_indices = label_indices[torch.randperm(total_data_points_with_label)]

        # Partitioning of the indices according to the Dirichlet distribution.
        partitioned_indices = list(torch.split(shuffled_indices, num_samples_per_partition))

        # Dropping the last partition as they are "excess" indices
        return partitioned_indices[:-1], min_samples, partition_allocations

    def partition_dataset(self, original_dataset: D, max_retries: int = 5) -> Tuple[List[D], Dict[T, np.ndarray]]:
        """
        Attempts partitioning of the original dataset up to max_retries times. Retries are potentially required if
        the user requests a minimum number of labels be assigned to each of the partitions. If the drawn Dirichlet
        distribution violates this minimum, a new distribution is drawn. This is repeated until the number of retries
        is exceeded or the minimum threshold is met.

        Args:
            original_dataset (D): The dataset to be partitioned
            max_retries (int, optional): Number of times to attempt to satisfy a user provided minimum
                label-associated data points per partition. Defaults to 5.

        Raises:
            ValueError: Throws this error if the retries have been exhausted and the user provided minimum is not met.

        Returns:
            List[D]: The partitioned datasets, length should correspond to self.number_of_partitions
            Dict[T, np.ndarray]: The Dirichlet distribution used to partition the data points for each label.
        """
        targets = original_dataset.targets
        assert targets is not None, "A label-based partitioner requires targets but this dataset has no targets"
        partitioned_indices = [torch.Tensor([]).int() for _ in range(self.number_of_partitions)]

        partition_attempts = 0
        partitioned_probabilities: Dict[T, np.ndarray] = {}
        for label in self.unique_labels:
            label_indices = torch.where(targets == label)[0].int()
            min_selected_labels = -1
            while min_selected_labels < self.min_label_examples:
                partitioned_indices_for_label, min_selected_labels, partitioned_probability = (
                    self.partition_label_indices(label, label_indices)
                )
                # If the minimum number of labels is satisfied or if there is a prior distribution, we accept the
                # partition. Otherwise, we retry.
                if self.prior_distribution is not None or min_selected_labels >= self.min_label_examples:
                    partitioned_probabilities[label] = partitioned_probability
                    for i, indices_for_label_partition in enumerate(partitioned_indices_for_label):
                        partitioned_indices[i] = torch.cat((partitioned_indices[i], indices_for_label_partition))
                else:
                    partition_attempts += 1
                    log(
                        INFO,
                        (
                            f"Too few datapoints in a partition. One partition had {min_selected_labels} but the "
                            f"minimum requested was {self.min_label_examples}. Resampling the partition..."
                        ),
                    )
                    if partition_attempts == max_retries:
                        raise ValueError(
                            (
                                f"Max Retries: {max_retries} reached. Partitioning failed to "
                                "satisfy the minimum label threshold"
                            )
                        )

        return [
            select_by_indices(original_dataset, indices) for indices in partitioned_indices
        ], partitioned_probabilities
