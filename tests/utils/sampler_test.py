import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from fl4health.utils.dataset import SyntheticDataset
from fl4health.utils.load_data import split_data_and_targets
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler, MinorityLabelBasedSampler


def construct_synthetic_dataset() -> tuple[SyntheticDataset, SyntheticDataset]:
    # set seed for creation
    torch.manual_seed(42)
    random_inputs = torch.rand((20000, 3, 3))
    # Note high is exclusive here
    random_labels = torch.randint(low=0, high=10, size=(20000, 1))
    train_inputs, train_labels, validation_inputs, validation_labels = split_data_and_targets(
        random_inputs, random_labels, validation_proportion=0.2
    )

    # unset seed thereafter
    torch.seed()
    return SyntheticDataset(train_inputs, train_labels), SyntheticDataset(validation_inputs, validation_labels)


def test_minority_sampler() -> None:
    minority_numbers = {1, 2, 3}
    downsampling_ratio = 0.1
    sampler = MinorityLabelBasedSampler(
        unique_labels=list(range(10)), downsampling_ratio=downsampling_ratio, minority_labels=minority_numbers
    )

    train_ds, val_ds = construct_synthetic_dataset()

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


def test_dirichlet_sampler_probability_assignment() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)

    # Very mild heterogeneity with beta 100
    sampler_1 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=100)
    prob_ratio_1 = min(sampler_1.probabilities) / max(sampler_1.probabilities)

    # Mild heterogeneity with beta 1
    sampler_2 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=10)
    prob_ratio_2 = min(sampler_2.probabilities) / max(sampler_2.probabilities)

    # Assert that the probabilities are more evenly distributed with higher beta
    assert prob_ratio_1 > prob_ratio_2

    # Severe heterogeneity with beta 0.1
    sampler_3 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.5)
    prob_ratio_3 = min(sampler_3.probabilities) / max(sampler_3.probabilities)

    # Assert that the probabilities are more evenly distributed with higher beta
    assert prob_ratio_2 > prob_ratio_3

    # Very severe heterogeneity with beta 0.0001
    sampler_4 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.001)
    prob_ratio_4 = min(sampler_4.probabilities) / max(sampler_4.probabilities)

    # Assert that the probabilities are more evenly distributed with higher beta
    assert prob_ratio_3 > prob_ratio_4

    unset_all_random_seeds()


def test_dirichlet_sampler_with_assigned_probability() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)
    sampler = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)

    train_ds, _ = construct_synthetic_dataset()

    # Training
    assert train_ds.targets is not None

    # No heterogeneity
    relative_probabilities = [1] * 10
    sampler.probabilities = np.array(relative_probabilities) / sum(relative_probabilities)

    train_new_ds = sampler.subsample(train_ds)
    assert train_new_ds.targets is not None

    new_samples_per_class = [train_new_ds.targets[train_new_ds.targets == i].size(0) for i in range(10)]
    new_probabilities = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    # Assert that probabilities of sampled indices are close to assigned probabilities
    for i in range(10):
        assert pytest.approx(new_probabilities[i], abs=0.001) == sampler.probabilities[i]

    # Mild heterogeneity
    relative_probabilities = list(range(1, 11))
    sampler.probabilities = np.array(relative_probabilities) / sum(relative_probabilities)

    train_new_ds = sampler.subsample(train_ds)
    assert train_new_ds.targets is not None

    new_samples_per_class = [train_new_ds.targets[train_new_ds.targets == i].size(0) for i in range(10)]
    new_probabilities = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    # Assert that probabilities of sampled indices are close to assigned probabilities
    for i in range(10):
        assert pytest.approx(new_probabilities[i], abs=0.001) == sampler.probabilities[i]

    # Extreme heterogeneity
    relative_probabilities = list(range(1, 101, 10))
    sampler.probabilities = np.array(relative_probabilities) / sum(relative_probabilities)
    train_new_ds = sampler.subsample(train_ds)
    assert train_new_ds.targets is not None

    new_samples_per_class = [train_new_ds.targets[train_new_ds.targets == i].size(0) for i in range(10)]
    new_probabilities = [x / sum(new_samples_per_class) for x in new_samples_per_class]

    # Assert that probabilities of sampled indices are close to assigned probabilities
    for i in range(10):
        assert pytest.approx(new_probabilities[i], abs=0.001) == sampler.probabilities[i]

    unset_all_random_seeds()


def test_dirichlet_sampler_without_hash_key() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    # Here we don't set the hash_key to have different random samplers
    sampler_1 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)
    sampler_2 = DirichletLabelBasedSampler(unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1)

    train_ds, _ = construct_synthetic_dataset()

    # Training
    assert train_ds.targets is not None

    samples_per_class = [train_ds.targets[train_ds.targets == i].size(0) for i in range(10)]

    train_new_ds_1 = sampler_1.subsample(train_ds)
    assert train_new_ds_1.targets is not None

    new_samples_per_class_1 = [train_new_ds_1.targets[train_new_ds_1.targets == i].size(0) for i in range(10)]

    train_new_ds_2 = sampler_2.subsample(train_ds)
    assert train_new_ds_2.targets is not None

    new_samples_per_class_2 = [train_new_ds_2.targets[train_new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_2, f_exp=new_samples_per_class_1)
    # Assert that the new distribution with sampler_1 is different from the original distribution
    assert p_val_1 < 0.01
    # Assert that the new distribution with sampler_2 is different from the original distribution
    assert p_val_2 < 0.01
    # Assert that the new distributions with sampler_1 and sampler_2 are different due to different random seeds
    assert p_val_3 < 0.01

    unset_all_random_seeds()


def test_dirichlet_sampler_with_hash_key() -> None:
    # Set the random seeds
    set_all_random_seeds(2023)
    # Kind of hacky way to ensure sampled label distribution differs from original label distribution
    # Here we set the hash_key to the same value for both samplers
    sampler_1 = DirichletLabelBasedSampler(
        unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1, hash_key=1000
    )
    sampler_2 = DirichletLabelBasedSampler(
        unique_labels=list(range(10)), sample_percentage=1.0, beta=0.1, hash_key=1000
    )

    train_ds, val_ds = construct_synthetic_dataset()

    # Training
    assert train_ds.targets is not None

    samples_per_class = [train_ds.targets[train_ds.targets == i].size(0) for i in range(10)]

    train_new_ds_1 = sampler_1.subsample(train_ds)
    assert train_new_ds_1.targets is not None

    new_samples_per_class_1 = [train_new_ds_1.targets[train_new_ds_1.targets == i].size(0) for i in range(10)]

    train_new_ds_2 = sampler_2.subsample(train_ds)
    assert train_new_ds_2.targets is not None

    new_samples_per_class_2 = [train_new_ds_2.targets[train_new_ds_2.targets == i].size(0) for i in range(10)]

    _, p_val_1 = chisquare(f_obs=new_samples_per_class_1, f_exp=samples_per_class)
    _, p_val_2 = chisquare(f_obs=new_samples_per_class_2, f_exp=samples_per_class)
    _, p_val_3 = chisquare(f_obs=new_samples_per_class_1, f_exp=new_samples_per_class_2)

    # Assert that the new distribution with sampler_1 is different from the original distribution
    assert p_val_1 < 0.01
    # Assert that the new distribution with sampler_2 is different from the original distribution
    assert p_val_2 < 0.01
    # Assert that the new distributions with sampler_1 and sampler_1 are same due to same hash_key
    assert p_val_3 == 1.0

    # We also want to make sure the hash key fixed the probability distribution for sampling and the sampling itself
    assert np.allclose(sampler_1.probabilities, sampler_2.probabilities, rtol=0.0, atol=1e-5)
    assert torch.allclose(train_new_ds_2.targets, train_new_ds_2.targets, rtol=0.0, atol=1e-5)
    assert torch.allclose(train_new_ds_2.data, train_new_ds_2.data, rtol=0.0, atol=1e-5)

    # Validation
    assert val_ds.targets is not None

    test_samples_per_class = [val_ds.targets[val_ds.targets == i].size(0) for i in range(10)]

    new_test_ds_1 = sampler_1.subsample(val_ds)

    assert new_test_ds_1.targets is not None

    new_test_samples_per_class_1 = [new_test_ds_1.targets[new_test_ds_1.targets == i].size(0) for i in range(10)]

    _, p_test_val_1 = chisquare(f_obs=test_samples_per_class, f_exp=new_test_samples_per_class_1)

    #  Assert that the new test distribution with sampler_1 is different from the original test distribution
    assert p_test_val_1 < 0.01

    train_probs = np.array([i / sum(samples_per_class) for i in samples_per_class])
    test_probs = np.array([i / sum(test_samples_per_class) for i in test_samples_per_class])
    # Assert that the original train and test distributions are same
    # atol is set to 1e-2 because there might be some rounding noise due to set fixed number of samples
    assert np.allclose(train_probs, test_probs, rtol=0.0, atol=1e-2)

    new_train_probs = np.array([i / sum(new_samples_per_class_1) for i in new_samples_per_class_1])
    new_test_probs = np.array([i / sum(new_test_samples_per_class_1) for i in new_test_samples_per_class_1])
    # Assert that the new train and test distributions with sampler_1 are same due to same hash_key
    # atol is set to 1e-2 because there might be some rounding noise due to set fixed number of samples
    assert np.allclose(new_train_probs, new_test_probs, rtol=0.0, atol=1e-2)
    # Assert that the new train distribution with sampler_1 is different from the original test distribution
    # atol is set to 1e-3 because there might be some rounding noise due to set fixed number of samples
    assert not np.allclose(new_train_probs, test_probs, rtol=0.0, atol=1e-3)
    # Assert that the new test distribution with sampler_1 is different from the original train distribution
    # atol is set to 1e-3 because there might be some rounding noise due to set fixed number of samples
    assert not np.allclose(train_probs, new_test_probs, rtol=0.0, atol=1e-3)

    unset_all_random_seeds()
