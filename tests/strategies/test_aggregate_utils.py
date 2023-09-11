import numpy as np
import pytest

from fl4health.strategies.aggregate_utils import aggregate_losses, aggregate_results


def test_aggregate_results() -> None:
    client_1_weights_samples = ([np.ones((2, 2, 2)), 3.0 * np.ones((2, 2, 3))], 3)
    # Modify first entry for thoroughness
    client_1_weights_samples[0][0][0, 0, 0] = 10

    client_2_weights_samples = ([np.ones((2, 2, 2)), np.ones((2, 2, 3))], 2)
    client_3_weights_samples = ([2.0 * np.ones((2, 2, 2)), np.ones((2, 2, 3))], 1)
    results = [client_1_weights_samples, client_2_weights_samples, client_3_weights_samples]

    # Aggregate using linearly weighted combination
    weighted_aggregate = aggregate_results(results, weighted=True)
    # Aggregate using uniform weighting
    unweighted_aggregate = aggregate_results(results, weighted=False)

    # Make sure they aren't equal
    assert not np.allclose(weighted_aggregate[0], unweighted_aggregate[0], atol=0.0001)

    # Weighted aggregate of the first layer should have entries 7/6 everywhere except the 0, 0, 0 entry
    # which should be 17/3. Weighted aggregate of the second layer should all be 2.0
    weighted_target_1 = np.ones((2, 2, 2)) * (7.0 / 6.0)
    weighted_target_1[0, 0, 0] = 17.0 / 3.0
    weighted_target_2 = np.ones((2, 2, 3)) * 2.0
    assert np.allclose(weighted_target_1, weighted_aggregate[0])
    assert np.allclose(weighted_target_2, weighted_aggregate[1])

    # Unweighted aggregate of the first layer should have entries 4/3 everywhere except the 0, 0, 0 entry
    # which should be 13/3. Weighted aggregate of the second layer should all be 4/3
    unweighted_target_1 = np.ones((2, 2, 2)) * (4.0 / 3.0)
    unweighted_target_1[0, 0, 0] = 13.0 / 3.0
    unweighted_target_2 = np.ones((2, 2, 3)) * (5.0 / 3.0)
    assert np.allclose(unweighted_target_1, unweighted_aggregate[0])
    assert np.allclose(unweighted_target_2, unweighted_aggregate[1])


def test_aggregate_losses() -> None:
    results = [(3, 1.0), (2, 2.0), (1, 10.0)]

    weighted_aggregate = aggregate_losses(results, weighted=True)
    unweighted_aggregate = aggregate_losses(results, weighted=False)

    assert pytest.approx(weighted_aggregate, abs=0.0001) == (1.0 * 3 + 2.0 * 2 + 10.0 * 1) / (6.0)
    assert pytest.approx(unweighted_aggregate, abs=0.0001) == (13.0) / (3.0)
