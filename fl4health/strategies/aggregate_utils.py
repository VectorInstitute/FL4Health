from functools import reduce

import numpy as np
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


def aggregate_results(results: list[tuple[NDArrays, int]], weighted: bool = True) -> NDArrays:
    """
    Compute weighted or unweighted average.

    Args:
        results (list[tuple[NDArrays, int]]): This is a set of ``NDArrays`` (list of numpy arrays) and the number of
            relevant samples from each client (training or validation samples where appropriate). These are to be
            aggregated together in a weighted or unweighted average. The ``NDArrays`` most often represent model
            states.
        weighted (bool, optional): Whether or not the aggregation is a weighted average (by the sample counts
            provided in the tuple) or a uniform average. Defaults to True.

    Returns:
        (NDArrays): Aggregated numpy arrays by the desired averaging.
    """
    if weighted:
        # Uses the underlying flwr aggregation scheme
        return aggregate(results)
    # Number of client weights to average
    num_clients = len(results)
    # Create a list of weights, each multiplied by 1/num_clients
    weighted_weights = [[layer * (1.0 / num_clients) for layer in weights] for weights, _ in results]

    # Compute unweighted average by summing up across clients for each layer.
    return [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]


def aggregate_losses(results: list[tuple[int, float]], weighted: bool = True) -> float:
    """
    Aggregate evaluation results obtained from multiple clients.

    Args:
        results (list[tuple[int, float]]): A list of sample counts and loss values (in that order). The sample counts
            from each client (training or validation samples where appropriate) are used if weighted averaging is
            requested.
        weighted (bool, optional): Whether or not the aggregation is a weighted average (by the sample counts
            provided in the tuple) or a uniform average. Defaults to True.

    Returns:
        (float): The weighted or unweighted average of the loss values in the results list.
    """
    # Sorting the results by the loss values for numerical fluctuation determinism of the sum
    results = sorted(results, key=lambda x: x[1])
    if weighted:
        # uses flwr implementation of weighted loss averaging
        return weighted_loss_avg(results)
    # standard averaging
    return sum([loss for _, loss in results]) / len(results)
