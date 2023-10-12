from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from fl4health.strategies.aggregate_utils import aggregate_losses, aggregate_results
from fl4health.strategies.basic_fedavg import BasicFedAvg


class FedProx(BasicFedAvg):
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
        proximal_weight: float,
        adaptive_proximal_weight: bool = False,
        proximal_weight_delta: float = 0.1,
        proximal_weight_patience: int = 5,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
        weighted_train_losses: bool = False,
    ) -> None:
        """
        A generalization of the fedavg strategy for FedProx. In addition to the model weights, the server also receives
        the training loss from the clients. If adaptive FedProx is enabled, these losses are used to update the
        proximal weight parameter according to the FedProx paper recommendations.

        The aggregation strategy for weights is the same as in FedAvg.

        Implementation based on https://arxiv.org/abs/1602.05629.

        Args:
            initial_parameters (Parameters): Initial global model parameters.
            proximal_weight (float): Initial proximal weight (mu). If adaptivity is false, then this is the constant
                weight used for all clients.
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): _description_. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system.
                Defaults to 2.
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
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            adaptive_proximal_weight (bool, optional): Determines whether the value of mu is adaptively modified by
                the server based on aggregated train loss. Defaults to False.
            proximal_weight_delta (float, optional): This is the amount by which the server changes the value of mu
                based on the modification criteria. Only applicable if adaptivity is on. Defaults to 0.1.
            proximal_weight_patience (int, optional): This is the number of rounds a server must see decreasing
                aggregated train loss before reducing the value of mu. Only applicable if adaptivity is on.
                Defaults to 5.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            weighted_train_losses (bool, optional): Determines whether the training losses from the clients should be
                aggregated using a weighted or unweighted average. These aggregated losses are used to adjust the
                proximal weight in the adaptive setting. Defaults to False.
        """

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
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
        self.parameter_packer = ParameterPackerFedProx()
        self.weighted_train_losses = weighted_train_losses

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the results from the federated fit round and, if applicable determine whether the proximal weight
        should be updated based on the aggregated loss seen on the clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In FedProx, the clients pack the weights to be
                aggregated along with the training loss seen during their local training cycle.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
                For FedProx, the server also packs a proximal weight to be sent to the clients. This is sent even if
                adaptive proximal weights are turned off and the value simply remains constant.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results with packed params of model weights and training loss
        weights_and_counts: List[Tuple[NDArrays, int]] = []
        train_losses_and_counts: List[Tuple[int, float]] = []
        for _, fit_res in results:
            sample_count = fit_res.num_examples
            updated_weights, train_loss = self.parameter_packer.unpack_parameters(
                parameters_to_ndarrays(fit_res.parameters)
            )
            weights_and_counts.append((updated_weights, sample_count))
            train_losses_and_counts.append((sample_count, train_loss))

        # Aggregate them in a weighted or unweighted fashion based on settings.
        weights_aggregated = aggregate_results(weights_and_counts, self.weighted_aggregation)

        # Aggregate train loss
        train_losses_aggregated = aggregate_losses(train_losses_and_counts, self.weighted_train_losses)

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
        """
        Update proximal weight parameter if adaptive_proximal_weight is set to True. Regardless of whether adaptivity
        is turned on at this time, the previous loss seen by the server is updated.

        Args:
            loss (float): This is the loss to which we compare the previous loss seen by the server. For FedProx this
            should be the aggregated training loss seen by each client participating in training.
        """

        if self.adaptive_proximal_weight:
            if loss <= self.previous_loss:
                self.proximal_weight_patience_counter += 1
                if self.proximal_weight_patience_counter == self.proximal_weight_patience:
                    self.proximal_weight -= self.proximal_weight_delta
                    self.proximal_weight = max(0.0, self.proximal_weight)
                    self.proximal_weight_patience_counter = 0
                    log(INFO, f"Aggregate training loss has dropped {self.proximal_weight_patience} rounds in a row")
                    log(INFO, f"Proximal weight is decreased to {self.proximal_weight}")
            else:
                self.proximal_weight += self.proximal_weight_delta
                self.proximal_weight_patience_counter = 0
                log(
                    INFO,
                    f"Aggregate training loss increased this round: Current loss {loss}, "
                    f"Previous loss: {self.previous_loss}",
                )
                log(INFO, f"Proximal weight is increased by {self.proximal_weight_delta} to {self.proximal_weight}")
        self.previous_loss = loss
