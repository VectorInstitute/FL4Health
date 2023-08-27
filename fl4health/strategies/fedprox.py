from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx


class FedProx(FedAvg):
    """
    A generalization of the fedavg strategy for fedprox
    Additional to the model weights, the server also receives the training loss from the clients,
    and updates the proximal weight parameter, accordingly.
    Aggregation strategy for weights is the same as in FedAvg.
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
        proximal_weight: float,
        adaptive_proximal_weight: bool = True,
        proximal_weight_delta: float = 0.1,
        proximal_weight_patience: int = 5,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        self.proximal_weight = proximal_weight
        self.adaptive_proximal_weight = adaptive_proximal_weight

        if self.adaptive_proximal_weight:
            self.proximal_weight_delta = proximal_weight_delta
            self.proximal_weight_patience = proximal_weight_patience
            self.proximal_weight_patience_counter: int = 0

        self.previous_loss = float("inf")

        self.server_model_weights = parameters_to_ndarrays(initial_parameters)
        initial_parameters.tensors.extend(ndarrays_to_parameters([np.array(proximal_weight)]).tensors)

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
        )
        self.parameter_packer = ParameterPackerFedProx()

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results with packed params of model weights and training loss
        weights_and_counts = []
        train_losses = []
        for _, fit_res in results:
            sample_count = fit_res.num_examples
            updated_weights, train_loss = self.parameter_packer.unpack_parameters(
                parameters_to_ndarrays(fit_res.parameters)
            )
            weights_and_counts.append((updated_weights, sample_count))
            train_losses.append(train_loss)

        # Aggregate model weights using fedavg aggregation strategy
        weights_aggregated = aggregate(weights_and_counts)

        # Aggregate train loss using unweighted average
        train_losses_aggregated = np.mean(train_losses)

        self._maybe_update_proximal_weight_param(float(train_losses_aggregated))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        parameters = self.parameter_packer.pack_parameters(weights_aggregated, self.proximal_weight)

        return ndarrays_to_parameters(parameters), metrics_aggregated

    def _maybe_update_proximal_weight_param(self, loss: float) -> None:

        """Update proximal weight parameter if adaptive_proximal_weight is set to True"""

        if self.adaptive_proximal_weight:
            if loss <= self.previous_loss:
                self.proximal_weight_patience_counter += 1
                if self.proximal_weight_patience_counter == self.proximal_weight_patience:
                    self.proximal_weight -= self.proximal_weight_delta
                    self.proximal_weight = max(0.0, self.proximal_weight)
                    self.proximal_weight_patience_counter = 0
                    log(INFO, f"Proximal weight is decreased to {self.proximal_weight}")
            else:
                self.proximal_weight += self.proximal_weight_delta
                self.proximal_weight_patience_counter = 0
                log(INFO, f"Proximal weight is increased to {self.proximal_weight}")
        self.previous_loss = loss
        return None
