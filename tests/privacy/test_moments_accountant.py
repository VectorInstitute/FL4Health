from math import floor

import pytest

from fl4health.privacy.moments_accountant import FixedSamplingWithoutReplacement, MomentsAccountant, PoissonSampling


def test_instance_accountant_reproduce_results() -> None:
    accountant = MomentsAccountant()
    # Vanilla DP-SGD test
    # This test "reproduces" the results of Figure 2 from Deep Learning with Differential Privacy.
    # The bounds are actually tighter than those of the paper due to an improvement in the sharpness of such bounds in
    # 2020 through https://arxiv.org/abs/2004.00010 Proposition 12 (in v4). See the documentation in the
    # rdp_privacy_accountant get_epsilon function. If you revert to the previous best bound of
    #   epsilon = min( rdp - math.log(delta) / (orders - 1) )
    # from https://arxiv.org/abs/1702.07476 Proposition 3 in v3 the results are reproduced exactly.

    # q = L/N
    sampling_ratio = 0.01
    # sigma
    noise_multiplier = 4.0
    # T=E/q
    updates = [10000, 40000]
    target_delta = 1 / pow(10, 5)
    expected_epsilons = [1.035, 2.213]
    for t, expected_epsilon in zip(updates, expected_epsilons):
        epsilon = accountant.get_epsilon(PoissonSampling(sampling_ratio), noise_multiplier, t, target_delta)
        assert pytest.approx(expected_epsilon, abs=0.01) == epsilon

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
    n_clients = 100
    clients_sampled_per_round = 10
    batch_size = 100
    client_data_points = 600
    server_rounds_list = [640, 288, 54]
    epochs_per_server_round = 1
    batch_steps = floor(client_data_points / batch_size)
    target_epsilons = [1.083, 1.109, 1.169]
    target_deltas = [9 / pow(10, 6), 9 / pow(10, 7), 1 / pow(10, 11)]
    noise_multipliers = [4.0, 3.0, 2.0]
    for server_rounds, z, delta, epsilon in zip(server_rounds_list, noise_multipliers, target_deltas, target_epsilons):
        population_size = n_clients * client_data_points
        sample_size = clients_sampled_per_round * batch_size
        t = server_rounds * epochs_per_server_round * batch_steps
        estimated_epsilon = accountant.get_epsilon(PoissonSampling(sample_size / population_size), z, t, delta)
        assert pytest.approx(epsilon, abs=0.001) == estimated_epsilon


def test_user_level_accountant_poisson_sampling_reproduce_results() -> None:
    """
    This test "reproduces" the results of Table 1 from Learning Differentially Private Recurrent Language Models.

    The bounds are actually tighter than those of the paper due to an improvement in the sharpness of such bounds in
    2020 through https://arxiv.org/abs/2004.00010 Proposition 12 (in v4). See the documentation in the
    ``rdp_privacy_accountant`` get_epsilon function. If you revert to the previous best bound of

    `#!python epsilon = min( rdp - math.log(delta) / (orders - 1) )`

    from https://arxiv.org/abs/1702.07476 Proposition 3 in v3 the results are reproduced exactly.
    """
    accountant = MomentsAccountant()
    noise_values = [1.0, 1.0, 1.0, 1.0, 3.0, 1.0]
    n_clients = [pow(10, 5), pow(10, 6), pow(10, 6), pow(10, 6), pow(10, 6), pow(10, 9)]
    clients_per_round = [pow(10, 2), pow(10, 1), pow(10, 3), pow(10, 4), pow(10, 3), pow(10, 3)]
    sampling_strategies = [PoissonSampling(c / n) for c, n in zip(clients_per_round, n_clients)]
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

    for k, c, z, strategy, d in zip(n_clients, clients_per_round, noise_values, sampling_strategies, target_deltas):
        expected_epsilons = expected_results[(k, c, z)]
        for t in updates:
            estimated_epsilon = accountant.get_epsilon(strategy, z, t, d)
            expected_epsilon = expected_epsilons[t]
            assert pytest.approx(expected_epsilon, abs=0.001) == estimated_epsilon


def test_user_level_accountant_with_equivalent_trajectories() -> None:
    # Tests whether performing the same process in a sequence of accountant processes is equivalent to a gathered
    # set of accounting
    accountant = MomentsAccountant()
    trajectory_length = 3
    noise_multiplier = 1.0
    noise_multipliers = [noise_multiplier] * trajectory_length
    sampling_rate = 0.2
    sampling_rates = [sampling_rate] * trajectory_length
    sampling_strategies = [PoissonSampling(q) for q in sampling_rates]
    updates = 10000
    updates_trajectory = [updates] * trajectory_length
    target_delta = 1 / pow(pow(10, 9), 1.1)

    trajectory_epsilon = accountant.get_epsilon(
        sampling_strategies, noise_multipliers, updates_trajectory, target_delta
    )

    non_trajectory_updates = updates * trajectory_length
    non_trajectory_epsilon = accountant.get_epsilon(
        PoissonSampling(sampling_rate), noise_multiplier, non_trajectory_updates, target_delta
    )

    assert pytest.approx(non_trajectory_epsilon, abs=0.01) == trajectory_epsilon


def test_user_level_accountant_with_longer_trajectories() -> None:
    # Tests whether performing the same process in a sequence of accountant processes is equivalent to a gathered
    # set of accounting
    accountant = MomentsAccountant()

    # increasing noise smaller epsilon
    smaller_epsilon = accountant.get_epsilon(
        [PoissonSampling(0.2)] * 3, [1.0, 1.2, 1.4], [10000] * 3, 1 / pow(pow(10, 9), 1.1)
    )
    compare_epsilon = accountant.get_epsilon(PoissonSampling(0.2), 1.0, 10000 * 3, 1 / pow(pow(10, 9), 1.1))

    assert smaller_epsilon < compare_epsilon

    # increasing updates, increase epsilon

    larger_epsilon = accountant.get_epsilon(
        [PoissonSampling(0.2)] * 3, [1.0] * 3, [10000, 12000, 14000], 1 / pow(pow(10, 9), 1.1)
    )
    compare_epsilon = accountant.get_epsilon(PoissonSampling(0.2), 1.0, 10000 * 3, 1 / pow(pow(10, 9), 1.1))

    assert larger_epsilon > compare_epsilon

    # increasing sample rates, increase epsilon

    larger_epsilon = accountant.get_epsilon(
        [PoissonSampling(0.2), PoissonSampling(0.3), PoissonSampling(0.4)],
        [1.0] * 3,
        [10000] * 3,
        1 / pow(pow(10, 9), 1.1),
    )
    compare_epsilon = accountant.get_epsilon(PoissonSampling(0.2), 1.0, 10000 * 3, 1 / pow(pow(10, 9), 1.1))

    assert larger_epsilon > compare_epsilon


def test_user_accountant_fixed_sampling_reproduce_results() -> None:
    # This test reproduces the results of Table 1 from Differentially private learning with Adaptive Clipping
    accountant = MomentsAccountant()
    noise_values = [0.669, 0.513, 0.659, 0.510, 1.396, 1.396]
    n_clients = 1000000
    clients_per_round = [2231, 513, 2197, 510, 13958, 13958]
    sampling_strategies = [
        FixedSamplingWithoutReplacement(n, c) for c, n in zip(clients_per_round, [n_clients] * len(clients_per_round))
    ]
    updates = [4000, 1500, 3000, 1200, 1500, 1500]
    # epsilon values should be close to this, for input of expected delta
    expected_epsilon = 5.0
    # delta values should be close to this value, for input of expected epsilon
    expected_delta = 1 / pow(n_clients, 1.1)
    for strategy, z, t in zip(sampling_strategies, noise_values, updates):
        estimated_epsilon = accountant.get_epsilon(strategy, z, t, expected_delta)
        assert pytest.approx(expected_epsilon, abs=0.1) == estimated_epsilon
        estimated_delta = accountant.get_delta(strategy, z, t, expected_epsilon)
        assert pytest.approx(expected_delta, abs=2 * pow(10, -8)) == estimated_delta
