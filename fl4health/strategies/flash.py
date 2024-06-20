from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fl4health.strategies.basic_fedavg import BasicFedAvg


class Flash(BasicFedAvg):
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
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
        weighted_aggregation: bool = False,
        weighted_eval_losses: bool = False,
    ) -> None:
        """Flash: Concept Drift Adaptation in Federated Learning.

        Implementation based on https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf

        Args:
            fraction_fit : float, optional
                Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate : float, optional
                Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients : int, optional
                Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients : int, optional
                Minimum number of clients used during validation. Defaults to 2.
            min_available_clients : int, optional
                Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
                          Optional[Tuple[float, Dict[str, Scalar]]]]]
                Optional function used for validation. Defaults to None.
            on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
                Function used to configure validation. Defaults to None.
            accept_failures : bool, optional
                Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters : Parameters
                Initial global model parameters.
            fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            eta : float, optional
                Server-side learning rate. Defaults to 1e-1.
            eta_l : float, optional
                Client-side learning rate. Defaults to 1e-1.
            beta_1 : float, optional
                Momentum parameter. Defaults to 0.9.
            beta_2 : float, optional
                Second moment parameter. Defaults to 0.99.
            tau : float, optional
                Controls the algorithm's degree of adaptability. Defaults to 1e-9.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. Flash default is a uniform average by the number of clients.
                Defaults to False.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. Flash default is a uniform average of the losses by dividing
                the total loss by the number of clients. Defaults to False.
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
        self.current_weights = parameters_to_ndarrays(initial_parameters)
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        zero_weights = [np.zeros_like(x) for x in self.current_weights]
        self.m_t: NDArrays = zero_weights.copy()
        self.v_t: NDArrays = zero_weights.copy()
        self.d_t: NDArrays = zero_weights.copy()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Flash(accept_failures={self.accept_failures})"
        return rep

    def _update_parameters(self, delta_t: NDArrays) -> None:
        """Update m_t, v_t, beta_3, and d_t per-element"""
        for i, (delta, m_prev, v_prev, d_prev) in enumerate(zip(delta_t, self.m_t, self.v_t, self.d_t)):
            delta_squared = np.square(delta)
            # Update m_t
            self.m_t[i] = self.beta_1 * m_prev + (1 - self.beta_1) * delta

            # Update v_t
            self.v_t[i] = self.beta_2 * v_prev + (1 - self.beta_2) * delta_squared

            # Compute beta_3
            norm_v_prev = np.abs(v_prev)
            norm_diff = np.abs(delta_squared - self.v_t[i])
            beta_3_matrix = norm_v_prev / (norm_diff + norm_v_prev)

            # Update d_t
            self.d_t[i] = beta_3_matrix * d_prev + (1 - beta_3_matrix) * (delta_squared - self.v_t[i])

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using the Flash method."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        delta_t: NDArrays = [x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)]

        self._update_parameters(delta_t)

        new_weights = [
            current_weight + self.eta * m_t / (np.sqrt(v_t) - d_t + self.tau)
            for current_weight, m_t, v_t, d_t in zip(self.current_weights, self.m_t, self.v_t, self.d_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
