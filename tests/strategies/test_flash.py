import numpy as np
from flwr.common import Code, FitRes, Metrics, NDArrays, Status, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.flash import Flash
from tests.test_utils.custom_client_proxy import CustomClientProxy


def construct_fit_res(parameters: NDArrays, metric: float, num_examples: int) -> FitRes:
    return FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(parameters),
        num_examples=num_examples,
        metrics={"metric": metric},
    )


client0_res = construct_fit_res([np.ones((3, 3)), np.ones((4, 4))], 0.1, 50)
client1_res = construct_fit_res([np.ones((3, 3)), np.full((4, 4), 2)], 0.2, 50)
client2_res = construct_fit_res([np.full((3, 3), 3), np.full((4, 4), 3)], 0.3, 100)
client3_res = construct_fit_res([np.full((3, 3), 4), np.full((4, 4), 4)], 0.4, 200)
clients_res_1: list[tuple[ClientProxy, FitRes]] = [
    (CustomClientProxy("c0"), client0_res),
    (CustomClientProxy("c1"), client1_res),
    (CustomClientProxy("c2"), client2_res),
    (CustomClientProxy("c3"), client3_res),
]

client0_res = construct_fit_res([np.full((3, 3), 1.5), np.full((4, 4), 1.5)], 0.15, 60)
client1_res = construct_fit_res([np.full((3, 3), 2.5), np.full((4, 4), 2.5)], 0.25, 60)
client2_res = construct_fit_res([np.full((3, 3), 3.5), np.full((4, 4), 3.5)], 0.35, 110)
client3_res = construct_fit_res([np.full((3, 3), 4.5), np.full((4, 4), 4.5)], 0.45, 210)
clients_res_2: list[tuple[ClientProxy, FitRes]] = [
    (CustomClientProxy("c0"), client0_res),
    (CustomClientProxy("c1"), client1_res),
    (CustomClientProxy("c2"), client2_res),
    (CustomClientProxy("c3"), client3_res),
]


def metrics_aggregation(to_aggregate: list[tuple[int, Metrics]]) -> Metrics:
    # Select last set of metrics (dummy for test)
    return to_aggregate[-1][1]


evaluate_metrics_aggregation_fn = metrics_aggregation
fit_metrics_aggregation_fn = metrics_aggregation

flash_strategy = Flash(
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    fraction_evaluate=0.8,
    fraction_fit=0.8,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    accept_failures=True,
    initial_parameters=ndarrays_to_parameters([np.zeros((3, 3)), np.zeros((4, 4))]),
    eta=0.1,
    eta_l=0.1,
    beta_1=0.9,
    beta_2=0.99,
    tau=1e-9,
)


def test_flash_aggregate_fit() -> None:
    # Expected m_t, v_t, d_t, beta_3 for round 1
    expected_m_t = [0.1 * (np.ones((3, 3)) * 2.25), 0.1 * (np.ones((4, 4)) * 2.5)]
    expected_v_t = [0.01 * (np.ones((3, 3)) * 2.25**2), 0.01 * (np.ones((4, 4)) * 2.5**2)]
    expected_d_t = [np.ones((3, 3)) * (2.25**2 - 0.050625), np.ones((4, 4)) * (2.5**2 - 0.0625)]

    # Perform the first round of updates
    flash_strategy.aggregate_fit(server_round=1, results=clients_res_1, failures=[])

    # Assertions for round 1
    assert flash_strategy.m_t is not None
    for mt, expected in zip(flash_strategy.m_t, expected_m_t):
        assert mt.shape == expected.shape, f"Round 1: Expected shape {expected.shape}, but got {mt.shape}"
        assert np.allclose(mt, expected), f"Round 1: Expected {expected}, but got {mt}"

    assert flash_strategy.v_t is not None
    for vt, expected in zip(flash_strategy.v_t, expected_v_t):
        assert vt.shape == expected.shape, f"Round 1: Expected shape {expected.shape}, but got {vt.shape}"
        assert np.allclose(vt, expected), f"Round 1: Expected {expected}, but got {vt}"

    assert flash_strategy.d_t is not None
    for dt, expected in zip(flash_strategy.d_t, expected_d_t):
        assert dt.shape == expected.shape, f"Round 1: Expected shape {expected.shape}, but got {dt.shape}"
        assert np.allclose(dt, expected), f"Round 1: Expected {expected}, but got {dt}"

    # Calculate new weights after round 1
    new_weights_round1 = [
        0.1 * expected_m_t[i] / (np.sqrt(expected_v_t[i]) - expected_d_t[i] + 1e-9) for i in range(len(expected_m_t))
    ]

    # Assertions for new weights after round 1
    for nw, expected in zip(flash_strategy.current_weights, new_weights_round1):
        assert nw.shape == expected.shape, f"Round 1: Expected shape {expected.shape}, but got {nw.shape}"
        assert np.allclose(nw, expected), f"Round 1: Expected new weights {expected}, but got {nw}"

    # Calculate delta_t for round 2
    new_aggregated_weights = [np.full((3, 3), 3.0), np.full((4, 4), 3.0)]
    delta_t_round2 = [w - nw for w, nw in zip(new_aggregated_weights, new_weights_round1)]

    # Expected m_t, v_t, d_t, beta_3 for round 2
    expected_m_t_round2 = [
        0.9 * expected_m_t[0] + 0.1 * delta_t_round2[0],
        0.9 * expected_m_t[1] + 0.1 * delta_t_round2[1],
    ]
    expected_v_t_round2 = [
        0.99 * expected_v_t[0] + 0.01 * (delta_t_round2[0] ** 2),
        0.99 * expected_v_t[1] + 0.01 * (delta_t_round2[1] ** 2),
    ]
    expected_beta_3_round2 = [
        0.050625 / (np.abs(delta_t_round2[0] ** 2 - expected_v_t_round2[0]) + 0.050625),
        0.0625 / (np.abs(delta_t_round2[1] ** 2 - expected_v_t_round2[1]) + 0.0625),
    ]

    expected_d_t_round2 = [
        expected_beta_3_round2[0] * expected_d_t[0]
        + (1 - expected_beta_3_round2[0]) * ((delta_t_round2[0] ** 2) - expected_v_t_round2[0]),
        expected_beta_3_round2[1] * expected_d_t[1]
        + (1 - expected_beta_3_round2[1]) * ((delta_t_round2[1] ** 2) - expected_v_t_round2[1]),
    ]

    # Perform the second round of updates
    flash_strategy.aggregate_fit(server_round=2, results=clients_res_2, failures=[])

    # Assertions for round 2
    assert flash_strategy.m_t is not None
    for mt, expected in zip(flash_strategy.m_t, expected_m_t_round2):
        assert mt.shape == expected.shape, f"Round 2: Expected shape {expected.shape}, but got {mt.shape}"
        assert np.allclose(mt, expected), f"Round 2: Expected {expected}, but got {mt}"

    assert flash_strategy.v_t is not None
    for vt, expected in zip(flash_strategy.v_t, expected_v_t_round2):
        assert vt.shape == expected.shape, f"Round 2: Expected shape {expected.shape}, but got {vt.shape}"
        assert np.allclose(vt, expected), f"Round 2: Expected {expected}, but got {vt}"

    assert flash_strategy.d_t is not None
    for dt, expected in zip(flash_strategy.d_t, expected_d_t_round2):
        assert dt.shape == expected.shape, f"Round 2: Expected shape {expected.shape}, but got {dt.shape}"
        assert np.allclose(dt, expected), f"Round 2: Expected {expected}, but got {dt}"

    # Calculate new weights after round 2
    new_weights_round2 = [
        new_weights_round1[i]
        + 0.1 * expected_m_t_round2[i] / (np.sqrt(expected_v_t_round2[i]) - expected_d_t_round2[i] + 1e-9)
        for i in range(len(expected_m_t_round2))
    ]

    # Assertions for new weights after round 2
    for nw, expected in zip(flash_strategy.current_weights, new_weights_round2):
        assert nw.shape == expected.shape, f"Round 2: Expected shape {expected.shape}, but got {nw.shape}"
        assert np.allclose(nw, expected), f"Round 2: Expected new weights {expected}, but got {nw}"
