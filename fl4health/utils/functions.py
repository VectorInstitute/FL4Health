from typing import Any

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import FitRes, NDArrays
from flwr.server.client_proxy import ClientProxy


class BernoulliSample(torch.autograd.Function):
    """
    Bernoulli sampling function that allows for gradient computation.

    Bernoulli sampling is by itself not differentiable, so in order to integrate it with autograd,
    this implementation follows the paper "Estimating or propagating gradients through stochastic neurons for
    conditional computation" and simply returns the Bernoulli probabilities themselves as the "gradient." This is
    called the "straight-through estimator." For more details, please see Section 4 of the aforementioned paper
    (https://arxiv.org/pdf/1308.3432).
    """

    @staticmethod
    def forward(bernoulli_probs: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(input=bernoulli_probs)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        assert len(inputs) == 1
        (bernoulli_probs,) = inputs
        ctx.save_for_backward(bernoulli_probs)

    # This method determines the "gradient" of the BernoulliSample function.
    # grad_output is supposed to be the gradient w.r.t. the output of the forward method.
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        # ctx.saved_tensors is a tuple (of length 1 in this case). Hence the indexing here.
        bernoulli_probs = ctx.saved_tensors[0]
        return bernoulli_probs * grad_output


bernoulli_sample = BernoulliSample.apply


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(1 / x - 1)


def select_zeroeth_element(array: np.ndarray) -> float:
    """
    Helper function that simply selects the first element of an array (index 0 across all dimensions).

    Args:
        array (np.ndarray): Array from which the very first element is selected.

    Returns:
        (float): zeroeth element value.
    """
    indices = tuple(0 for _ in array.shape)
    return array[indices]


def pseudo_sort_scoring_function(client_result: tuple[ClientProxy, NDArrays, int]) -> float:
    """
    This function provides the "score" that is used to sort a list of ``tuple[ClientProxy, NDArrays, int]``. We select
    the zeroeth (index 0 across all dimensions) element from each of the arrays in the ``NDArrays`` list, sum them, and
    add the integer (client sample counts) to the sum to come up with a score for sorting. Note that the underlying
    numpy arrays in ``NDArrays`` may not all be of numerical type. So we limit to selecting elements from arrays of
    floats.

    Args:
        client_result (tuple[ClientProxy, NDArrays, int]]): Elements to use to determine the score.

    Returns:
        (float): Sum of a the zeroeth elements of each array in the ``NDArrays`` and the int of the tuple.
    """
    _, client_arrays, sample_count = client_result
    zeroeth_params = [
        select_zeroeth_element(array) for array in client_arrays if np.issubdtype(array.dtype, np.floating)
    ]
    return np.sum(zeroeth_params) + sample_count


def decode_and_pseudo_sort_results(
    results: list[tuple[ClientProxy, FitRes]],
) -> list[tuple[ClientProxy, NDArrays, int]]:
    """
    This function is used to convert the results of client training into ``NDArrays`` and to apply a pseudo sort
    based on the zeroeth elements in the weights and the sample counts. As long as the numpy seed has been set on the
    server this process should be deterministic when repeatedly running the same server code leading to deterministic
    sorting (assuming the clients are deterministically training their weights as well). This allows, for example,
    for weights from the clients to be summed in a deterministic order during aggregation.

    **NOTE**: Client proxies would be nice to use for this task, but the CIDs are set by uuid deep in the flower
    library and are, therefore, not pinnable without a ton of work.

    Args:
        results (list[tuple[ClientProxy, FitRes]]): Results from a federated training round.

    Returns:
        (list[tuple[ClientProxy, NDArrays, int]]): The ordered set of weights as ``NDarrays`` and the corresponding
        number of examples.
    """
    ndarrays_results = [
        (client_proxy, parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for client_proxy, fit_res in results
    ]
    return sorted(ndarrays_results, key=lambda x: pseudo_sort_scoring_function(x))
