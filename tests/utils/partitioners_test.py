import numpy as np
import pytest
import torch

from fl4health.utils.dataset import SyntheticDataset
from fl4health.utils.partitioners import DirichletLabelBasedAllocation


def construct_synthetic_dataset() -> SyntheticDataset:
    # set seed for creation
    torch.manual_seed(42)
    random_inputs = torch.rand((10000, 3, 3))
    # Note high is exclusive here
    random_labels = torch.randint(low=0, high=10, size=(10000, 1))
    # unset seed thereafter
    torch.seed()
    return SyntheticDataset(random_inputs, random_labels)


SYNTHETIC_DATASET = construct_synthetic_dataset()


def test_dirichlet_allocation_partitioner() -> None:
    # setting numpy seed for reproducibility
    np.random.seed(42)
    # Should be a fairly uniform partitioner with a large beta
    uniform_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=5, unique_labels=list(range(10)), beta=100.0
    )
    partitioned_datasets, _ = uniform_partitioner.partition_dataset(SYNTHETIC_DATASET)

    assert len(partitioned_datasets) == 5
    partition_0_targets = partitioned_datasets[0].targets
    partition_2_targets = partitioned_datasets[2].targets
    partition_4_targets = partitioned_datasets[4].targets

    assert partition_0_targets is not None
    assert partition_2_targets is not None
    assert partition_4_targets is not None

    assert len(torch.where(partition_0_targets == 0)[0]) == 203
    assert len(torch.where(partition_0_targets == 1)[0]) == 211
    assert len(torch.where(partition_2_targets == 5)[0]) == 203
    assert len(torch.where(partition_2_targets == 9)[0]) == 183
    assert len(torch.where(partition_4_targets == 0)[0]) == 226
    assert len(torch.where(partition_4_targets == 5)[0]) == 232

    # Should be a skewed partitioner with a small beta
    heterogeneous_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=10, unique_labels=list(range(10)), beta=1.0, min_label_examples=2
    )
    partitioned_datasets, _ = heterogeneous_partitioner.partition_dataset(SYNTHETIC_DATASET, max_retries=5)

    assert len(partitioned_datasets) == 10
    partition_0_targets = partitioned_datasets[0].targets
    partition_2_targets = partitioned_datasets[2].targets
    partition_6_targets = partitioned_datasets[6].targets
    partition_7_targets = partitioned_datasets[7].targets
    partition_9_targets = partitioned_datasets[9].targets

    assert partition_0_targets is not None
    assert partition_2_targets is not None
    assert partition_6_targets is not None
    assert partition_7_targets is not None
    assert partition_9_targets is not None

    assert len(partition_0_targets) == 943
    assert len(partition_6_targets) == 846
    assert len(partition_9_targets) == 816

    assert len(torch.where(partition_0_targets == 0)[0]) == 25
    assert len(torch.where(partition_0_targets == 1)[0]) == 180
    assert len(torch.where(partition_0_targets == 5)[0]) == 7
    assert len(torch.where(partition_0_targets == 9)[0]) == 243

    assert len(torch.where(partition_2_targets == 0)[0]) == 34
    assert len(torch.where(partition_2_targets == 1)[0]) == 133
    assert len(torch.where(partition_2_targets == 5)[0]) == 200
    assert len(torch.where(partition_2_targets == 9)[0]) == 173

    assert len(torch.where(partition_7_targets == 0)[0]) == 204
    assert len(torch.where(partition_7_targets == 1)[0]) == 45
    assert len(torch.where(partition_7_targets == 5)[0]) == 33
    assert len(torch.where(partition_7_targets == 8)[0]) == 27

    # unset seed for safety
    np.random.seed()


def test_dirichlet_allocation_partitioner_with_prior_distribution() -> None:
    # setting numpy seed for reproducibility
    np.random.seed(42)
    number_of_partitions = 5
    unique_labels = list(range(10))
    prior_distribution = {
        label: np.array([partition + label + 1 for partition in range(number_of_partitions)])
        for label in unique_labels
    }
    # Partitioner with a given prior distribution
    prior_based_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=number_of_partitions, unique_labels=unique_labels, prior_distribution=prior_distribution
    )
    partitioned_datasets, _ = prior_based_partitioner.partition_dataset(SYNTHETIC_DATASET)

    assert len(partitioned_datasets) == 5
    partition_0_targets = partitioned_datasets[0].targets
    partition_2_targets = partitioned_datasets[2].targets
    partition_4_targets = partitioned_datasets[4].targets

    assert partition_0_targets is not None
    assert partition_2_targets is not None
    assert partition_4_targets is not None

    assert len(torch.where(partition_0_targets == 0)[0]) == 66
    assert len(torch.where(partition_0_targets == 1)[0]) == 96
    assert len(torch.where(partition_2_targets == 5)[0]) == 211
    assert len(torch.where(partition_2_targets == 9)[0]) == 194
    assert len(torch.where(partition_4_targets == 0)[0]) == 333
    assert len(torch.where(partition_4_targets == 5)[0]) == 263

    # Test passing output of the first partitioner as prior distribution to the second partitioner
    first_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=10, unique_labels=list(range(10)), beta=1.0, min_label_examples=2
    )
    partitioned_datasets_1, partitioned_distribution_1 = first_partitioner.partition_dataset(
        SYNTHETIC_DATASET, max_retries=5
    )

    assert len(partitioned_datasets_1) == 10
    partition_0_targets_1 = partitioned_datasets_1[0].targets
    partition_2_targets_1 = partitioned_datasets_1[2].targets
    partition_6_targets_1 = partitioned_datasets_1[6].targets
    partition_7_targets_1 = partitioned_datasets_1[7].targets
    partition_9_targets_1 = partitioned_datasets_1[9].targets

    assert partition_0_targets_1 is not None
    assert partition_2_targets_1 is not None
    assert partition_6_targets_1 is not None
    assert partition_7_targets_1 is not None
    assert partition_9_targets_1 is not None

    second_partitioner = DirichletLabelBasedAllocation(
        number_of_partitions=10, unique_labels=list(range(10)), prior_distribution=partitioned_distribution_1
    )

    partitioned_datasets_2, partitioned_distribution_2 = second_partitioner.partition_dataset(
        SYNTHETIC_DATASET, max_retries=5
    )

    # Check that the partitioned distributions are the same
    for key in partitioned_distribution_1:
        for partition in range(10):
            assert (
                pytest.approx(partitioned_distribution_1[key][partition], abs=0.00001)
                == partitioned_distribution_2[key][partition]
            )

    assert len(partitioned_datasets_2) == 10
    partition_0_targets_2 = partitioned_datasets_2[0].targets
    partition_2_targets_2 = partitioned_datasets_2[2].targets
    partition_6_targets_2 = partitioned_datasets_2[6].targets
    partition_7_targets_2 = partitioned_datasets_2[7].targets
    partition_9_targets_2 = partitioned_datasets_2[9].targets

    assert partition_0_targets_2 is not None
    assert partition_2_targets_2 is not None
    assert partition_6_targets_2 is not None
    assert partition_7_targets_2 is not None
    assert partition_9_targets_2 is not None

    assert len(partition_0_targets_2) == len(partition_0_targets_1)
    assert len(partition_2_targets_2) == len(partition_2_targets_1)
    assert len(partition_6_targets_2) == len(partition_6_targets_1)
    assert len(partition_7_targets_2) == len(partition_7_targets_1)
    assert len(partition_9_targets_2) == len(partition_9_targets_1)

    for i in range(10):
        assert len(torch.where(partition_0_targets_2 == i)[0]) == len(torch.where(partition_0_targets_1 == i)[0])
        assert len(torch.where(partition_2_targets_2 == i)[0]) == len(torch.where(partition_2_targets_1 == i)[0])
        assert len(torch.where(partition_6_targets_2 == i)[0]) == len(torch.where(partition_6_targets_1 == i)[0])
        assert len(torch.where(partition_7_targets_2 == i)[0]) == len(torch.where(partition_7_targets_1 == i)[0])
        assert len(torch.where(partition_9_targets_2 == i)[0]) == len(torch.where(partition_9_targets_1 == i)[0])

    # unset seed for safety
    np.random.seed()
