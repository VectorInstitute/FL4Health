import numpy as np
import torch
from flwr.common import NDArray, NDArrays

from fl4health.strategies.fedavg_sparse_coo_tensor import FedAvgSparseCooTensor


client1_tensor_names = ["tensor1", "tensor2"]
client2_tensor_names = ["tensor2", "tensor3"]
client3_tensor_names = ["tensor3", "tensor4"]
client_train_sizes = [25, 75, 100]
total_train_size = sum(client_train_sizes)


def create_coo_tensor_diagonal(n: int, all_ones: bool) -> tuple[NDArray, NDArray, NDArray]:
    parameters = np.array([1 for _ in range(1, n + 1)]) if all_ones else np.array(list(range(1, n + 1)))
    indices = np.array([[j, j] for j in range(n)])
    shape = np.array([n, n])

    return parameters, indices, shape


def create_client_parameters(
    num_tensors: int, sizes: list[int], all_ones_lst: list[bool]
) -> tuple[NDArrays, NDArrays, NDArrays]:
    assert len(sizes) == num_tensors and len(sizes) == len(all_ones_lst)
    client_parameters = []
    client_indices = []
    client_shapes = []

    for _, n, all_ones in zip(range(num_tensors), sizes, all_ones_lst):
        parameters, indices, shape = create_coo_tensor_diagonal(n, all_ones)

        client_parameters.append(parameters)
        client_indices.append(indices)
        client_shapes.append(shape)
    return client_parameters, client_indices, client_shapes


def test_aggregate() -> None:
    client1_sizes = [3, 4]
    all_ones_lst = [True, False]
    client2_sizes = [4, 5]
    client3_sizes = [5, 6]

    client1_parameters, client1_indices, client1_shapes = create_client_parameters(2, client1_sizes, all_ones_lst)

    client2_parameters, client2_indices, client2_shapes = create_client_parameters(2, client2_sizes, all_ones_lst)

    client3_parameters, client3_indices, client3_shapes = create_client_parameters(2, client3_sizes, all_ones_lst)

    strategy = FedAvgSparseCooTensor(weighted_aggregation=True)

    # Construct the weights results in the correct format.
    client1_res = strategy.parameter_packer.pack_parameters(
        client1_parameters, (client1_indices, client1_shapes, client1_tensor_names)
    )

    client2_res = strategy.parameter_packer.pack_parameters(
        client2_parameters, (client2_indices, client2_shapes, client2_tensor_names)
    )

    client3_res = strategy.parameter_packer.pack_parameters(
        client3_parameters, (client3_indices, client3_shapes, client3_tensor_names)
    )

    weights_results = [
        (client1_res, client_train_sizes[0]),
        (client2_res, client_train_sizes[1]),
        (client3_res, client_train_sizes[2]),
    ]

    aggregated_results = strategy.aggregate(weights_results)

    expected_results = {
        "tensor1": torch.diag(torch.tensor([1, 1, 1])),
        "tensor2": torch.diag(
            (torch.tensor([1, 2, 3, 4]) * client_train_sizes[0] + torch.tensor([1, 1, 1, 1]) * client_train_sizes[1])
            / (client_train_sizes[0] + client_train_sizes[1])
        ),
        "tensor3": torch.diag(
            (
                torch.tensor([1, 2, 3, 4, 5]) * client_train_sizes[1]
                + torch.tensor([1, 1, 1, 1, 1]) * client_train_sizes[2]
            )
            / (client_train_sizes[1] + client_train_sizes[2])
        ),
        "tensor4": torch.diag(torch.tensor([1, 2, 3, 4, 5, 6])),
    }

    assert expected_results.keys() == aggregated_results.keys()
    for key, expected_val in expected_results.items():
        assert (expected_val == aggregated_results[key]).all()
