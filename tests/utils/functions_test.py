import numpy as np
import pytest
import torch
from flwr.common import Code, Status, ndarrays_to_parameters
from flwr.common.typing import FitRes, NDArrays
from flwr.server.client_proxy import ClientProxy

from fl4health.utils.functions import (
    bernoulli_sample,
    decode_and_pseudo_sort_results,
    pseudo_sort_scoring_function,
    select_zeroeth_element,
    sigmoid_inverse,
)
from tests.test_utils.custom_client_proxy import CustomClientProxy


def test_bernoulli_gradient() -> None:
    torch.manual_seed(42)
    theta = torch.rand(7)
    theta.requires_grad = True
    pred = bernoulli_sample(theta)
    target = torch.ones(7)
    loss = torch.sum((pred - target) ** 2)
    loss.backward()
    assert (theta.grad == 2 * (pred - target) * theta).all()
    torch.seed()


def test_sigmoid_inverse() -> None:
    torch.manual_seed(42)
    x = torch.rand(7)
    z = torch.sigmoid(x)
    assert torch.allclose(sigmoid_inverse(z), x)
    torch.seed()


def test_select_zeroeth_element() -> None:
    np.random.seed(42)
    array = np.random.rand(10, 10)
    random_element = select_zeroeth_element(array)
    assert pytest.approx(random_element, abs=1e-5) == 0.3745401188473625
    np.random.seed(None)


def test_pseudo_sort_scoring_function() -> None:
    np.random.seed(42)
    array_list = [np.random.rand(10, 10) for _ in range(2)] + [np.random.rand(5, 5) for _ in range(2)]
    sort_value = pseudo_sort_scoring_function((CustomClientProxy("c0"), array_list, 13))
    assert pytest.approx(sort_value, abs=1e-5) == 14.291990594067467
    np.random.seed(None)


def test_pseudo_sort_scoring_function_with_mixed_types() -> None:
    np.random.seed(42)
    array_list = (
        [np.random.rand(10, 10) for _ in range(2)]
        + [np.array(["Cat", "Dog"]), np.array([True, False])]
        + [np.random.rand(5, 5) for _ in range(2)]
    )
    sort_value = pseudo_sort_scoring_function((CustomClientProxy("c0"), array_list, 13))
    assert pytest.approx(sort_value, abs=1e-5) == 14.291990594067467
    np.random.seed(None)


def construct_fit_res(parameters: NDArrays, metric: float, num_examples: int) -> FitRes:
    return FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(parameters),
        num_examples=num_examples,
        metrics={"metric": metric},
    )


def test_decode_and_pseudo_sort_results() -> None:
    np.random.seed(42)
    client0_res = construct_fit_res([np.ones((3, 3)), np.ones((4, 4))], 0.1, 100)
    client1_res = construct_fit_res([np.ones((3, 3)), np.full((4, 4), 2.0)], 0.2, 75)
    client2_res = construct_fit_res([np.full((3, 3), 3.0), np.full((4, 4), 3.0)], 0.3, 50)
    clients_res: list[tuple[ClientProxy, FitRes]] = [
        (CustomClientProxy("c0"), client0_res),
        (CustomClientProxy("c1"), client1_res),
        (CustomClientProxy("c2"), client2_res),
    ]

    sorted_results = decode_and_pseudo_sort_results(clients_res)
    assert sorted_results[0][2] == 50
    assert sorted_results[1][2] == 75
    assert sorted_results[2][2] == 100

    np.random.seed(None)
