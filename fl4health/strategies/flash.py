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
    """Flash - Concept Drift Adaptation in Federated Learning.

    Implementation based on https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf

    Parameters
    ----------
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
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
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
    d_t : NDArrays, optional
        Drift-aware term, initialized to None and updated during aggregation.
        Helps adjust the effective learning rate to quickly adapt to concept drifts.

    """

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
        self.beta_3: Optional[NDArrays] = None
        self.m_t: Optional[NDArrays] = None
        self.v_t: Optional[NDArrays] = None
        self.d_t: Optional[NDArrays] = None

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Flash(accept_failures={self.accept_failures})"
        return rep

    def _update_d_t(self, delta_t: NDArrays, beta_3: NDArrays) -> None:
        """Update the drift-aware term d_t."""
        assert self.v_t is not None and self.d_t is not None
        for i, (delta, v_prev, d_prev) in enumerate(zip(delta_t, self.v_t, self.d_t)):
            d_t_j = []
            for j in range(len(delta)):
                d_t_j.append(beta_3[i][j] * d_prev[j] + (1 - beta_3[i][j]) * ((delta[j] ** 2) - v_prev[j]))
            self.d_t[i] = np.array(d_t_j)

    def _update_beta_3(self, delta_t: NDArrays, v_t_prev: NDArrays) -> NDArrays:
        """Update the beta_3 term."""
        assert self.v_t is not None and v_t_prev is not None
        beta_3 = []
        for delta, v, v_prev in zip(delta_t, self.v_t, v_t_prev):
            beta_3_j = []
            for j in range(len(delta)):
                norm_v_prev = np.linalg.norm(v_prev[j])
                norm_diff = np.linalg.norm((delta[j] ** 2) - v[j])
                beta_3_j.append(norm_v_prev / (norm_diff + norm_v_prev))
            beta_3.append(np.array(beta_3_j))
        return beta_3

    def _update_v_t(self, delta_t: NDArrays) -> None:
        """Update the second moment estimate v_t."""
        assert self.v_t is not None
        self.v_t = [self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y) for x, y in zip(self.v_t, delta_t)]

    def _update_m_t(self, delta_t: NDArrays) -> None:
        """Update the first moment estimate m_t."""
        assert self.m_t is not None
        self.m_t = [np.multiply(self.beta_1, x) + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t)]

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
        for attr in ["m_t", "v_t", "d_t"]:
            if getattr(self, attr) is None:
                setattr(self, attr, [np.zeros_like(x) for x in delta_t])
        assert self.m_t is not None and self.v_t is not None and self.d_t is not None
        # m_t
        self._update_m_t(delta_t)

        # v_t
        v_t_prev = self.v_t
        self._update_v_t(delta_t)

        # d_t
        beta_3 = self._update_beta_3(delta_t, v_t_prev)
        self._update_d_t(delta_t, beta_3)

        # Update global weights
        new_weights = [
            current_weight + self.eta * m_t / (np.sqrt(v_t) - d_t + self.tau)
            for current_weight, m_t, v_t, d_t in zip(self.current_weights, self.m_t, self.v_t, self.d_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
