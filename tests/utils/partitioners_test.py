import numpy as np
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
    partitioned_datasets = uniform_partitioner.partition_dataset(SYNTHETIC_DATASET)

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
    partitioned_datasets = heterogeneous_partitioner.partition_dataset(SYNTHETIC_DATASET, max_retries=5)

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
