from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
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
from flwr.server.strategy import Strategy

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.strategies.aggregate_utils import aggregate_results


class ModelMergeStrategy(Strategy):
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
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_aggregation: bool = True
    ) -> None:
        """
        Model Merging strategy in which weights are loaded from clients, averaged (weighted or unweighted)
            and redistributed to the clients for evaluation.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. In case `min_fit_clients` is
                larger than `fraction_fit * available_clients`, `min_fit_clients` will still be sampled.
                Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. In case
                `min_evaluate_clients` is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
                will still be sampled. Defaults to 1.0.
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
                counts. Defaults to True.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. Important to note that weighting is based on number of samples in the
                test dataset for the ModelMergeStrategy. Defaults to True.
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.weighted_aggregation = weighted_aggregation

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sample and configure clients for a fit round.

        In ModelMergeStrategy, it is assumed that server side parameters are empty and clients will
            be initialized with their weights locally.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): Not used.
            client_manager (ClientManager): The manager used to sample from the available clients.

        Returns:
            List[Tuple[ClientProxy, FitIns]]: List of sampled client identifiers and the configuration/parameters to
                be sent to each client (packaged as FitIns).
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(Parameters([], ""), config)

        # Sample clients
        if isinstance(client_manager, BaseFractionSamplingManager):
            clients = client_manager.sample_fraction(self.fraction_fit, self.min_available_clients)
        else:
            sample_size = max(int(client_manager.num_available() * self.fraction_fit), self.min_fit_clients)
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Sample and configure clients for a evaluation round.

        Args:
            server_round (int): Indicates the server round we're currently on. Only one round for ModelMergeStrategy
            parameters (Parameters): The parameters to be used to initialize the clients for the eval round.
                This will only occur following model merging.
            client_manager (ClientManager): The manager used to sample from the available clients.

        Returns:
            List[Tuple[ClientProxy, EvaluateIns]]: List of sampled client identifiers and the configuration/parameters
                to be sent to each client (packaged as EvaluateIns).
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if isinstance(client_manager, BaseFractionSamplingManager):
            clients = client_manager.sample_fraction(self.fraction_evaluate, self.min_available_clients)
        else:
            sample_size = max(int(client_manager.num_available() * self.fraction_evaluate), self.min_evaluate_clients)
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Performs model merging by taking an unweighted average of client weights and metrics.

        Args:
            server_round (int): Indicates the server round we're currently on. Only one round for ModelMergeStrategy.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local fit
                that need to be aggregated on the server-side.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during fit, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]
        # Aggregate them in an weighted or unweighted fashion based on self.weighted_aggregation.
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
        """
        Aggregate the metrics returned from the clients as a result of the evaluation round.
            ModelMergeStrategy assumes only metrics will be computed on client and loss is set to None.

        Args:
            results (List[Tuple[ClientProxy, EvaluateRes]]): The client identifiers and the results of their local
                evaluation that need to be aggregated on the server-side. These results are loss values
                (None in this case) and the metrics dictionary.
            failures (List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]): These are the results and
                exceptions from clients that experienced an issue during evaluation, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[float], Dict[str, Scalar]]: Aggregated loss values and the aggregated metrics. The metrics
                are aggregated according to evaluate_metrics_aggregation_fn.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return None, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate the model parameters after the merging has occured. This function can be used to perform centralized
            (i.e., server-side) evaluation of model parameters.

        Args:
            server_round (int): Server round. Only one round in ModelMergeStrategy.
            parameters: Parameters The current model parameters after merging has occurred.

        Returns:
            Optional[Tuple[float, Dict[str, Scalar]]]: A Tuple containing loss and a
                dictionary containing task-specific metrics (e.g., accuracy).
        """
        if self.evaluate_fn is None:
            return None

        eval_res = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})

        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """
        Required definition of parent class. ModelMergeStrategy does not support server side initialization.
            Parameters are always set to None

        Args:
            client_manager (ClientManager): Unused.

        Returns:
            None
        """
        return None
