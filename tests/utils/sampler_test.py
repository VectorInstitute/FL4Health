from pathlib import Path

from scipy.stats import chisquare

from fl4health.utils.dataset import MNISTDataset
from fl4health.utils.sampler import DirichletLabelBasedSampler, MinorityLabelBasedSampler


def test_minority_sampler() -> None:
    minority_numbers = set([1, 2, 3])
    downsampling_ratio = 0.1
    sampler = MinorityLabelBasedSampler(
        unique_labels=list(range(10)), downsampling_ratio=downsampling_ratio, minority_labels=minority_numbers
    )

    # Training
    ds = MNISTDataset(data_path=Path("examples/datasets/MNIST"), train=True)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    ds_new = sampler.subsample(ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(ds_new) == sum(new_samples_per_class)

    # Testing
    ds = MNISTDataset(data_path=Path("examples/datasets/MNIST"), train=False)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    ds_new = sampler.subsample(ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(ds_new) == sum(new_samples_per_class)


def test_dirichlet_sampler() -> None:
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    sampler = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0)

    # Training
    ds = MNISTDataset(data_path=Path("examples/datasets/MNIST"), train=True)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds = sampler.subsample(ds)
    new_samples_per_class = [new_ds.targets[new_ds.targets == i].size(0) for i in range(10)]

    _, p_val = chisquare(f_obs=new_samples_per_class, f_exp=samples_per_class)

    # Assert that the new distribution is different from the orginal distribution
    assert p_val < 0.01

    # Testing
    ds = MNISTDataset(data_path=Path("examples/datasets/MNIST"), train=False)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds = sampler.subsample(ds)
    new_samples_per_class = [new_ds.targets[new_ds.targets == i].size(0) for i in range(10)]

    _, p_val = chisquare(f_obs=new_samples_per_class, f_exp=samples_per_class)

    # Assert that the new distribution is different from the orginal distribution
    assert p_val < 0.01
