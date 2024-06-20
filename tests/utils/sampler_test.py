from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chisquare

from fl4health.utils.dataset import MnistDataset
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler, MinorityLabelBasedSampler


def test_minority_sampler() -> None:
    minority_numbers = set([1, 2, 3])
    downsampling_ratio = 0.1
    sampler = MinorityLabelBasedSampler(
        unique_labels=list(range(10)), downsampling_ratio=downsampling_ratio, minority_labels=minority_numbers
    )

    # Training
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=True)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    ds_new = sampler.subsample(ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(ds_new) == sum(new_samples_per_class)

    # Testing
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=False)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    ds_new = sampler.subsample(ds)

    new_samples_per_class = [
        int(num_samples * downsampling_ratio) if i in minority_numbers else num_samples
        for i, num_samples in enumerate(samples_per_class)
    ]

    assert len(ds_new) == sum(new_samples_per_class)


def test_dirichlet_sampler_probability_assignment() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)

    # Very mildly heterogeneous with beta 100
    sampler_1 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=100)
    prob_ratio_1 = min(sampler_1.probabilities) / max(sampler_1.probabilities)

    # Mild heterogeneous with beta 1
    sampler_2 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=10)
    prob_ratio_2 = min(sampler_2.probabilities) / max(sampler_2.probabilities)

    assert prob_ratio_1 > prob_ratio_2

    # Sever heterogeneous with beta 0.1
    sampler_3 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.5)
    prob_ratio_3 = min(sampler_3.probabilities) / max(sampler_3.probabilities)

    assert prob_ratio_2 > prob_ratio_3

    # Very sever heterogeneous with beta 0.0001
    sampler_4 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.001)
    prob_ratio_4 = min(sampler_4.probabilities) / max(sampler_4.probabilities)

    assert prob_ratio_3 > prob_ratio_4

    unset_all_random_seeds()


def test_dirichlet_sampler_with_assigned_probability() -> None:

    set_all_random_seeds(2023)
    sampler = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=True)

    # Not heterogeneous
    relative_probs = [1] * 10
    sampler.probabilities = np.array([i for i in relative_probs]) / sum([i for i in relative_probs])

    new_ds = sampler.subsample(ds)
    new_samples_per_class = [new_ds.targets[new_ds.targets == i].size(0) for i in range(10)]
    new_probs = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    for i in range(10):
        assert pytest.approx(new_probs[i], abs=0.001) == sampler.probabilities[i]

    # Mild heterogeneous
    relative_probs = [i for i in range(1, 11)]
    sampler.probabilities = np.array([i for i in relative_probs]) / sum([i for i in relative_probs])

    new_ds = sampler.subsample(ds)
    new_samples_per_class = [new_ds.targets[new_ds.targets == i].size(0) for i in range(10)]
    new_probs = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    for i in range(10):
        assert pytest.approx(new_probs[i], abs=0.001) == sampler.probabilities[i]

    # Extreme heterogeneous
    relative_probs = [i for i in range(1, 101, 10)]
    sampler.probabilities = np.array([i for i in relative_probs]) / sum([i for i in relative_probs])
    new_ds = sampler.subsample(ds)
    new_samples_per_class = [new_ds.targets[new_ds.targets == i].size(0) for i in range(10)]
    new_probs = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    for i in range(10):
        assert pytest.approx(new_probs[i], abs=0.001) == sampler.probabilities[i]

    unset_all_random_seeds()


def test_dirichlet_sampler_without_hash_key() -> None:
    set_all_random_seeds(2023)
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    # Here we don't set the hash_key to have different random samplers
    sampler_1 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)
    sampler_2 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)

    # Training
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=True)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds_1 = sampler_1.subsample(ds)
    new_samples_per_class_1 = [new_ds_1.targets[new_ds_1.targets == i].size(0) for i in range(10)]

    new_ds_2 = sampler_2.subsample(ds)
    new_samples_per_class_2 = [new_ds_2.targets[new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_1, f_exp=new_samples_per_class_2)

    # Assert that the new distribution is different from the original distribution
    assert p_val_1 < 0.01
    assert p_val_2 < 0.01
    assert p_val_3 < 0.01

    # Testing
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=False)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds_1 = sampler_1.subsample(ds)
    new_samples_per_class_1 = [new_ds_1.targets[new_ds_1.targets == i].size(0) for i in range(10)]

    new_ds_2 = sampler_2.subsample(ds)
    new_samples_per_class_2 = [new_ds_2.targets[new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_1, f_exp=new_samples_per_class_2)

    # Assert that the new distribution is different from the original distribution
    assert p_val_1 < 0.01
    assert p_val_2 < 0.01
    assert p_val_3 < 0.01

    unset_all_random_seeds()


def test_dirichlet_sampler_with_hash_key() -> None:
    set_all_random_seeds(2023)
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    # Here we set the hash_key to the same value for both samplers
    sampler_1 = DirichletLabelBasedSampler(
        unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1, hash_key=1000
    )
    sampler_2 = DirichletLabelBasedSampler(
        unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1, hash_key=1000
    )

    # Training
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=True)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds_1 = sampler_1.subsample(ds)
    new_samples_per_class_1 = [new_ds_1.targets[new_ds_1.targets == i].size(0) for i in range(10)]

    new_ds_2 = sampler_2.subsample(ds)
    new_samples_per_class_2 = [new_ds_2.targets[new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_1, f_exp=new_samples_per_class_2)

    # Assert that the new distribution is different from the original distribution
    assert p_val_1 < 0.01
    assert p_val_2 < 0.01
    assert p_val_3 == 1.0

    # Testing
    ds = MnistDataset(data_path=Path("examples/datasets/MNIST"), train=False)
    samples_per_class = [ds.targets[ds.targets == i].size(0) for i in range(10)]

    new_ds_1 = sampler_1.subsample(ds)
    new_samples_per_class_1 = [new_ds_1.targets[new_ds_1.targets == i].size(0) for i in range(10)]

    new_ds_2 = sampler_2.subsample(ds)
    new_samples_per_class_2 = [new_ds_2.targets[new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_1, f_exp=new_samples_per_class_2)

    # Assert that the new distribution is different from the original distribution
    assert p_val_1 < 0.01
    assert p_val_2 < 0.01
    assert p_val_3 == 1.0

    unset_all_random_seeds()
