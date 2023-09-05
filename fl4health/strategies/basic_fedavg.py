"""Federated Averaging with Flexible Sampling
This implementation extends that of Flower in two ways. The first is that it provides an option for unweighted
averaging, where Flower only offers weighted averaging based on client sample counts. The second is that it allows
users to Flower's standard sampling or use a custom sampling approach implemented in by a custom client manager.

Paper: https://arxiv.org/abs/1602.05629
"""

from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.strategies.aggregate_utils import aggregate_losses, aggregate_results
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class BasicFedAvg(FedAvg, StrategyWithPolling):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
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
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        weighted_aggregation: bool, Optional.
            Defaults to True, determines whether parameter aggregation is a linearly weighted average or a uniform
            average. FedAvg default is weighted average by client dataset counts.
        weighted_eval_losses: bool, Optional
            Defaults to True, determines whether losses during evaluation are linearly weighted averages or a uniform
            average. FedAvg default is weighted average of the losses by client dataset counts.
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
        )
        self.weighted_aggregation = weighted_aggregation
        self.weighted_eval_losses = weighted_eval_losses

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        if isinstance(client_manager, BaseFractionSamplingManager):
            # Using one of the custom FractionSamplingManager classes, sampling fraction is based on fraction_fit
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)
            fit_ins = FitIns(parameters, config)

            # Sample clients
            clients = client_manager.sample_fraction(self.fraction_fit, self.min_available_clients)

            # Return client/config pairs
            return [(client, fit_ins) for client in clients]
        else:
            log(INFO, f"Using the standard Flower ClientManager: {type(client_manager)}")
            return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        if isinstance(client_manager, BaseFractionSamplingManager):
            # Using one of the custom FractionSamplingManager classes, sampling fraction is based on fraction_evaluate
            # Parameters and config
            config = {}
            if self.on_evaluate_config_fn is not None:
                # Custom evaluation config function provided
                config = self.on_evaluate_config_fn(server_round)
            evaluate_ins = EvaluateIns(parameters, config)

            # Sample clients
            clients = client_manager.sample_fraction(self.fraction_evaluate, self.min_available_clients)

            # Return client/config pairs
            return [(client, evaluate_ins) for client in clients]
        else:
            log(INFO, f"Using the standard Flower ClientManager: {type(client_manager)}")
            return super().configure_evaluate(server_round, parameters, client_manager)

    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        """Configure server for polling of clients."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        property_ins = GetPropertiesIns(config)

        if isinstance(client_manager, BaseFractionSamplingManager):
            clients = client_manager.sample_all(min_num_clients=self.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = client_manager.num_available()
            clients = client_manager.sample(num_available_clients, min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, property_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted or unweighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]
        # Aggregate them in a weighted or unweighted fashion based on settings.
        aggregated_arrays = aggregate_results(weights_results, self.weighted_aggregation)
        # Convert back to parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_arrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Get losses and number of examples from the evaluation results.
        loss_results = [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        # Then aggregate the losses
        loss_aggregated = aggregate_losses(loss_results, self.weighted_eval_losses)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
