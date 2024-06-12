import subprocess
import sys
from typing import Any

import pytest

from fl4health.privacy.fl_accountants import (
    FlClientLevelAccountantFixedSamplingNoReplacement,
    FlClientLevelAccountantPoissonSampling,
    FlInstanceLevelAccountant,
)


def pip_list() -> Any:
    args = [sys.executable, "-m", "pip", "list"]
    p = subprocess.run(args, check=True, capture_output=True)
    return p.stdout.decode()


def test_instance_accountant_varying_sizes() -> None:
    # Test whether the Instance Level FL with DP-SGD accountant handles varying client batch sizes and dataset sizes
    client_sampling_rate = 0.1
    client_batch_sizes = [200, 100]
    client_data_points = [700, 600]
    server_rounds_list = [640, 288, 54]
    epochs_per_server_round = 1
    target_epsilons = [1.083, 1.109, 1.169]
    target_deltas = [9 / pow(10, 6), 9 / pow(10, 7), 1 / pow(10, 11)]
    noise_multipliers = [4.0, 3.0, 2.0]
    for server_rounds, z, delta, epsilon in zip(server_rounds_list, noise_multipliers, target_deltas, target_epsilons):
        accountant = FlInstanceLevelAccountant(
            client_sampling_rate, z, epochs_per_server_round, client_batch_sizes, client_data_points
        )
        estimated_epsilon = accountant.get_epsilon(server_rounds, delta)
        assert epsilon < estimated_epsilon


def test_instance_accountant_reproduce_results() -> None:
    # Instance Level FL with DP-SGD
    # Vanilla DP-SGD test
    # This test "reproduces" the results of Table 4.3 from Differentially Private Federated Learning.
    # The bounds are actually tighter than those of the paper due to an improvement in the sharpness of such bounds in
    # 2020 through https://arxiv.org/abs/2004.00010 Proposition 12 (in v4). See the documentation in the
    # rdp_privacy_accountant get_epsilon function. If you revert to the previous best bound of
    #   epsilon = min( rdp - math.log(delta) / (orders - 1) )
    # from https://arxiv.org/abs/1702.07476 Proposition 3 in v3 the results are reproduced exactly.
    # NOTE: There is also an ERROR in that thesis where they claim the bounds for Fix sampling without replacement,
    # but use Poisson Sampling in there bound calculations.
    client_sampling_rate = 0.1
    client_batch_sizes = [100]
    client_data_points = [600]
    server_rounds_list = [640, 288, 54]
    epochs_per_server_round = 1
    target_epsilons = [1.083, 1.109, 1.169]
    target_deltas = [9 / pow(10, 6), 9 / pow(10, 7), 1 / pow(10, 11)]
    noise_multipliers = [4.0, 3.0, 2.0]
    for server_rounds, z, delta, epsilon in zip(server_rounds_list, noise_multipliers, target_deltas, target_epsilons):
        accountant = FlInstanceLevelAccountant(
            client_sampling_rate, z, epochs_per_server_round, client_batch_sizes, client_data_points
        )
        estimated_epsilon = accountant.get_epsilon(server_rounds, delta)
        assert pytest.approx(epsilon, abs=0.001) == estimated_epsilon


def test_user_level_accountant_poisson_sampling_reproduce_results() -> None:
    # This test "reproduces" the results of Table 1 from Learning Differentially Private Recurrent Language Models.
    # The bounds are actually tighter than those of the paper due to an improvement in the sharpness of such bounds in
    # 2020 through https://arxiv.org/abs/2004.00010 Proposition 12 (in v4). See the documentation in the
    # rdp_privacy_accountant get_epsilon function. If you revert to the previous best bound of
    #   epsilon = min( rdp - math.log(delta) / (orders - 1) )
    # from https://arxiv.org/abs/1702.07476 Proposition 3 in v3 the results are reproduced exactly.

    print(pip_list())
    assert 1 == 0


def test_user_level_accountant_with_equivalent_trajectories() -> None:
    # Tests whether performing the same process in a sequence of accountant processes is equivalent to a gathered
    # set of accounting
    trajectory_length = 3
    noise_multiplier = 1.0
    noise_multipliers = [noise_multiplier] * trajectory_length
    sampling_rate = 0.2
    sampling_rates = [sampling_rate] * trajectory_length
    updates = 10000
    updates_trajectory = [updates] * trajectory_length
    target_delta = 1 / pow(pow(10, 9), 1.1)

    trajectory_accountant = FlClientLevelAccountantPoissonSampling(sampling_rates, noise_multipliers)
    non_trajectory_accountant = FlClientLevelAccountantPoissonSampling(sampling_rate, noise_multiplier)

    trajectory_epsilon = trajectory_accountant.get_epsilon(updates_trajectory, target_delta)

    non_trajectory_updates = updates * trajectory_length
    non_trajectory_epsilon = non_trajectory_accountant.get_epsilon(non_trajectory_updates, target_delta)

    assert pytest.approx(non_trajectory_epsilon, abs=0.01) == trajectory_epsilon


def test_user_level_accountant_with_longer_trajectories() -> None:
    # Tests whether performing the same process in a sequence of accountant processes is equivalent to a gathered
    # set of accounting
    increasing_noise_accountant = FlClientLevelAccountantPoissonSampling([0.2] * 3, [1.0, 1.2, 1.4])
    constant_accountant = FlClientLevelAccountantPoissonSampling([0.2] * 3, [1.0] * 3)
    increasing_sample_rate_accountant = FlClientLevelAccountantPoissonSampling([0.2, 0.3, 0.4], [1.0] * 3)

    # increasing noise smaller epsilon
    smaller_epsilon = increasing_noise_accountant.get_epsilon([10000] * 3, 1 / pow(pow(10, 9), 1.1))
    compare_epsilon = constant_accountant.get_epsilon([10000] * 3, 1 / pow(pow(10, 9), 1.1))

    assert smaller_epsilon < compare_epsilon

    # increasing updates, increase epsilon

    larger_epsilon = constant_accountant.get_epsilon([10000, 12000, 14000], 1 / pow(pow(10, 9), 1.1))
    compare_epsilon = constant_accountant.get_epsilon([10000] * 3, 1 / pow(pow(10, 9), 1.1))

    assert larger_epsilon > compare_epsilon

    # increasing sample rates, increase epsilon

    larger_epsilon = increasing_sample_rate_accountant.get_epsilon([10000] * 3, 1 / pow(pow(10, 9), 1.1))
    compare_epsilon = constant_accountant.get_epsilon([10000] * 3, 1 / pow(pow(10, 9), 1.1))

    assert larger_epsilon > compare_epsilon


def test_user_accountant_fixed_sampling_reproduce_results() -> None:
    # This test reproduces the results of Table 1 from Differentially private learning with Adaptive Clipping

    noise_values = [0.669, 0.513, 0.659, 0.510, 1.396, 1.396]
    n_clients = 1000000
    clients_per_round = [2231, 513, 2197, 510, 13958, 13958]
    updates = [4000, 1500, 3000, 1200, 1500, 1500]
    # epsilon values should be close to this, for input of expected delta
    expected_epsilon = 5.0
    # delta values should be close to this value, for input of expected epsilon
    expected_delta = 1 / pow(n_clients, 1.1)
    for n_clients_per_round, z, t in zip(clients_per_round, noise_values, updates):
        accountant = FlClientLevelAccountantFixedSamplingNoReplacement(n_clients, n_clients_per_round, z)
        estimated_epsilon = accountant.get_epsilon(t, expected_delta)
        assert pytest.approx(expected_epsilon, abs=0.1) == estimated_epsilon
        estimated_delta = accountant.get_delta(t, expected_epsilon)
        assert pytest.approx(expected_delta, abs=2 * pow(10, -8)) == estimated_delta
