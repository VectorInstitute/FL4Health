import pytest

from fl4health.privacy.fl_accountants import (
    FlClientLevelAccountantFixedSamplingNoReplacement,
    FlClientLevelAccountantPoissonSampling,
    FlInstanceLevelAccountant,
)


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

    noise_values = [1.0, 1.0, 1.0, 1.0, 3.0, 1.0]
    n_clients = [pow(10, 5), pow(10, 6), pow(10, 6), pow(10, 6), pow(10, 6), pow(10, 9)]
    clients_per_round = [pow(10, 2), pow(10, 1), pow(10, 3), pow(10, 4), pow(10, 3), pow(10, 3)]
    sampling_probabilities = [c / n for c, n in zip(clients_per_round, n_clients)]
    updates = [1, 10, 100, 1000, 10000, 100000, 1000000]
    target_deltas = [1 / (pow(K, 1.1)) for K in n_clients]

    expected_results = {
        (n_clients[0], clients_per_round[0], noise_values[0]): {
            updates[0]: 0.697,
            updates[1]: 0.700,
            updates[2]: 0.725,
            updates[3]: 0.774,
            updates[4]: 0.884,
            updates[5]: 1.899,
            updates[6]: 6.830,
        },
        (n_clients[1], clients_per_round[1], noise_values[1]): {
            updates[0]: 0.504,
            updates[1]: 0.504,
            updates[2]: 0.504,
            updates[3]: 0.504,
            updates[4]: 0.507,
            updates[5]: 0.530,
            updates[6]: 0.532,
        },
        (n_clients[2], clients_per_round[2], noise_values[2]): {
            updates[0]: 0.892,
            updates[1]: 0.895,
            updates[2]: 0.919,
            updates[3]: 0.985,
            updates[4]: 1.095,
            updates[5]: 2.130,
            updates[6]: 7.505,
        },
        (n_clients[3], clients_per_round[3], noise_values[3]): {
            updates[0]: 1.366,
            updates[1]: 1.525,
            updates[2]: 1.685,
            updates[3]: 2.634,
            updates[4]: 7.810,
            updates[5]: 30.388,
            updates[6]: 160.853,
        },
        (n_clients[4], clients_per_round[4], noise_values[4]): {
            updates[0]: 0.162,
            updates[1]: 0.162,
            updates[2]: 0.163,
            updates[3]: 0.166,
            updates[4]: 0.200,
            updates[5]: 0.502,
            updates[6]: 1.705,
        },
        (n_clients[5], clients_per_round[5], noise_values[5]): {
            updates[0]: 0.684,
            updates[1]: 0.685,
            updates[2]: 0.685,
            updates[3]: 0.690,
            updates[4]: 0.712,
            updates[5]: 0.712,
            updates[6]: 0.712,
        },
    }

    for k, c, z, q, d in zip(n_clients, clients_per_round, noise_values, sampling_probabilities, target_deltas):
        expected_epsilons = expected_results[(k, c, z)]
        for t in updates:
            accountant = FlClientLevelAccountantPoissonSampling(q, z)
            estimated_epsilon = accountant.get_epsilon(t, d)
            expected_epsilon = expected_epsilons[t]
            assert pytest.approx(expected_epsilon, abs=0.001) == estimated_epsilon


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
