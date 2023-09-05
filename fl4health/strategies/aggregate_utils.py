from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


def aggregate_results(results: List[Tuple[NDArrays, int]], weighted: bool = True) -> NDArrays:
    """Compute weighted or unweighted average."""
    if weighted:
        return aggregate(results)
    else:
        # Number of client weights to average
        num_clients = len(results)
        # Create a list of weights, each multiplied by 1/num_clients
        weighted_weights = [[layer * (1.0 / num_clients) for layer in weights] for weights, _ in results]

        # Compute unweighted average by summing up across clients for each layer.
        return [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]


def aggregate_losses(results: List[Tuple[int, float]], weighted: bool = True) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    if weighted:
        return weighted_loss_avg(results)
    else:
        return sum([loss for _, loss in results]) / len(results)
