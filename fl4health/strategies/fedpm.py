from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import (
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
)
from flwr.common.typing import Scalar
from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer


class FedPm(FedAvgDynamicLayer):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_aggregation: bool = False,
        weighted_eval_losses: bool = True,
        bayesian_aggregation: bool = True,
    ) -> None:
        """
        A generalization of the FedAvg strategy where the server can receive any arbitrary subset of the layers from
        any arbitrary subset of the clients, and weighted average for each received layer is performed independently.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during fitting. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            evaluate_fn (Optional[
                Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
            ]):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure server-side central validation by providing a Config dictionary.
                Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Optional[Parameters], optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            bayesian_aggregation (bool): Determines whether Bayesian aggregation is used.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
        self.weighted_aggregation = False
        # Parameters for Bayesian aggregation
        self.beta_parameters: Dict[str, Tuple[NDArray, NDArray]] = {}
        self.bayesian_aggregation = bayesian_aggregation
    
    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        if not self.bayesian_aggregation:
            return super().aggregate(results)
        else:
            return self.aggregate_bayesian(results)
    
    def aggregate_bayesian(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        names_to_layers: DefaultDict[str, List[NDArray]] = defaultdict(list)
        total_num_clients: DefaultDict[str, int] = defaultdict(int)

        # unpack the parameters and initialize the beta parameters to be all ones if necessary.
        for packed_layers, _ in results:
            layers, names = self.parameter_packer.unpack_parameters(packed_layers)
            for layer, name in zip(layers, names):
                names_to_layers[name].append(layer)
                total_num_clients[name] += 1
                if name not in self.beta_parameters:
                    alpha = np.ones(shape=layer.shape)
                    beta = np.ones(shape=layer.shape)
                    self.beta_parameters[name] = (alpha, beta)
                    
        aggregation_result: Dict[str, NDArray] = {}
        
        # posterior update of the beta parameters and using them
        # to compute the final result.
        for parameter_name in self.beta_parameters.keys():
            m_agg = np.sum(names_to_layers[parameter_name])
            n_clients = total_num_clients[parameter_name]
            alpha, beta = self.beta_parameters[parameter_name]
            alpha_new = alpha + m_agg
            beta_new = beta + np.ones(beta.shape) * n_clients - m_agg
            self.beta_parameters[parameter_name] = (alpha_new, beta_new)
            aggregation_result[parameter_name] = (alpha_new - 1) / (alpha_new + beta_new - 2)
        
        return aggregation_result
    
    def reset_beta_priors(self) -> None:
        assert self.beta_parameters != {}
        for parameter_name in self.beta_parameters.keys():
            alpha, beta = self.beta_parameters[parameter_name]
            self.beta_parameters[parameter_name] = (alpha * 0 + 1, beta * 0 + 1)
