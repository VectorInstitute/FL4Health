from typing import Any, List, Tuple

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays
from flwr.common.typing import FitRes, NDArrays
from flwr.server.client_proxy import ClientProxy


class BernoulliSample(torch.autograd.Function):
    """
    Bernoulli sampling function that allows for gradient computation.

    Bernoulli sampling is by itself not differentiable, so in order to integrate it with autograd,
    this implementation follows the paper
    "Estimating or propagating gradients through stochastic neurons for conditional computation"
    and simply returns the Bernoulli probabilities themselves as the "gradient". This is called the
    "straight-through estimator". For more details, please see Section 4 of the aforementioned paper
    (https://arxiv.org/pdf/1308.3432).
    """

    @staticmethod
    def forward(bernoulli_probs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.bernoulli(input=bernoulli_probs)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        assert len(inputs) == 1
        (bernoulli_probs,) = inputs
        ctx.save_for_backward(bernoulli_probs)

    # This method determines the "gradient" of the BernoulliSample function.
    # grad_output is supposed to be the gradient w.r.t. the output of the forward method.
    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        # ctx.saved_tensors is a tuple (of length 1 in this case). Hence the indexing here.
        bernoulli_probs = ctx.saved_tensors[0]
        return bernoulli_probs * grad_output


bernoulli_sample = BernoulliSample.apply


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(1 / x - 1)


def select_random_element(array: np.ndarray) -> float:
    """
    Helper function that simply selects a random element from the provided array using np.random.choice.

    Args:
        array (np.ndarray): Array from which a single element is randomly selected

    Returns:
        float: randomly selected element value.
    """
    indices = tuple(np.random.choice(size) for size in array.shape)
    return array[indices]


def pseudo_sort_function(client_result: Tuple[ClientProxy, NDArrays, int]) -> float:
    """
    This function provides the "score" that is used to sort a list of Tuple[NDArrays, int]. We take a random selection
    of the array elements and add the integer to them to come up with a score for sorting.

    Args:
        client_result (Tuple[ClientProxy, NDArrays, int]]): Elements to use to determine the score.

    Returns:
        float: Sum of a random selection of a single element of each array in the NDArrays list and the int of the
        tuple
    """
    _, client_arrays, sample_count = client_result
    random_params = [
        select_random_element(array) for array in client_arrays if np.issubdtype(array.dtype, np.floating)
    ]
    return np.sum(random_params) + sample_count


def decode_and_pseudo_sort_results(
    results: List[Tuple[ClientProxy, FitRes]]
) -> List[Tuple[ClientProxy, NDArrays, int]]:
    """
    This function is used to convert the results of client training into NDArrays and to apply a pseudo sort
    based on a random selection of the elements in the weights. As long as the numpy seed has been set on the server
    this process should be deterministic when repeatedly running the same server code leading to deterministic sorting
    (assuming the clients are deterministically training their weights as well). This allows, for example, for weights
    from the clients to be summed in a deterministic order during aggregation.

    NOTE: Client proxies would be nice to use for this task, but the CIDs are set by uuid deep in the flower library
    and are, therefore, not pinnable without a ton of work.

    Args:
        results (List[Tuple[ClientProxy, FitRes]]): Results from a federated training round.

    Returns:
        List[Tuple[NDArrays, int]]: The ordered set of weights as NDarrays and the corresponding number of examples
            NOTE: We have ditched the client proxies.
    """
    ndarrays_results = [
        (client_proxy, parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for client_proxy, fit_res in results
    ]
    return sorted(ndarrays_results, key=lambda x: pseudo_sort_function(x))
