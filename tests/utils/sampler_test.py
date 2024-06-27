from pathlib import Path

from scipy.stats import chisquare

from fl4health.utils.load_data import get_train_and_val_mnist_datasets
from fl4health.utils.sampler import DirichletLabelBasedSampler, MinorityLabelBasedSampler


def test_minority_sampler() -> None:
    minority_numbers = set([1, 2, 3])
    downsampling_ratio = 0.1
    sampler = MinorityLabelBasedSampler(
        unique_labels=list(range(10)), downsampling_ratio=downsampling_ratio, minority_labels=minority_numbers
    )

    train_ds, val_ds = get_train_and_val_mnist_datasets(Path("examples/datasets/MNIST"))

    # Training
    assert train_ds.targets is not None

    samples_per_class = [train_ds.targets[train_ds.targets == i].size(0) for i in range(10)]

    train_ds_new = sampler.subsample(train_ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(train_ds_new) == sum(new_samples_per_class)

    # Validation
    assert val_ds.targets is not None

    samples_per_class = [val_ds.targets[val_ds.targets == i].size(0) for i in range(10)]

    val_ds_new = sampler.subsample(val_ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(val_ds_new) == sum(new_samples_per_class)


def test_dirichlet_sampler() -> None:
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    sampler = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)

    train_ds, val_ds = get_train_and_val_mnist_datasets(Path("examples/datasets/MNIST"))

    # Training
    assert train_ds.targets is not None

    samples_per_class = [train_ds.targets[train_ds.targets == i].size(0) for i in range(10)]

    train_new_ds = sampler.subsample(train_ds)

    assert train_new_ds.targets is not None

    new_samples_per_class = [train_new_ds.targets[train_new_ds.targets == i].size(0) for i in range(10)]

    _, p_val = chisquare(f_obs=new_samples_per_class, f_exp=samples_per_class)

    # Assert that the new distribution is different from the original distribution
    assert p_val < 0.01

    # Validation
    assert val_ds.targets is not None

    samples_per_class = [val_ds.targets[val_ds.targets == i].size(0) for i in range(10)]

    val_new_ds = sampler.subsample(val_ds)

    assert val_new_ds.targets is not None

    new_samples_per_class = [val_new_ds.targets[val_new_ds.targets == i].size(0) for i in range(10)]

    _, p_val = chisquare(f_obs=new_samples_per_class, f_exp=samples_per_class)

    # Assert that the new distribution is different from the original distribution
    assert p_val < 0.01
