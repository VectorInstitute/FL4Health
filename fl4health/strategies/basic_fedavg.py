from collections.abc import Callable
from logging import INFO, WARNING

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
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from opacus import GradSampleModule

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.strategies.aggregate_utils import aggregate_losses, aggregate_results
from fl4health.strategies.strategy_with_poll import StrategyWithPolling
from fl4health.utils.functions import decode_and_pseudo_sort_results
from fl4health.utils.parameter_extraction import get_all_model_parameters


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
        evaluate_fn: (
            Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        """
        Federated Averaging with Flexible Sampling. This implementation extends that of Flower in two ways. The first
        is that it provides an option for unweighted averaging, where Flower only offers weighted averaging based on
        client sample counts. The second is that it allows users to Flower's standard sampling or use a custom
        sampling approach implemented in by a custom client manager.

        FedAvg Paper: https://arxiv.org/abs/1602.05629.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. In case ``min_fit_clients`` is
                larger than ``fraction_fit * available_clients``, ``min_fit_clients`` will still be sampled.
                Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. In case
                ``min_evaluate_clients`` is larger than ``fraction_evaluate * available_clients``,
                ``min_evaluate_clients`` will still be sampled. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                client-side validation by providing a ``Config`` dictionary. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Parameters | None, optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
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

    def add_auxiliary_information(self, original_parameters: Parameters) -> None:
        """
        Identity function for the BasicFedAvg strategy. This function is made available for override in more complex
        FL strategies to allow for the strategies to add auxiliary information to sets of parameters. This function
        is specifically designed to allow addition to parameters initialized by the server calling out to a client for
        weight initialization.

        Here we need not add anything. So no modifications are made.

        Args:
            original_parameters (Parameters): Original set of parameters
        """
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        This function configures a sample of clients for a training round. It handles the case where the client
        manager has a sample fraction vs. a sample function (to allow for more flexible sampling).
        The function follows the standard configuration flow where the ``on_fit_config_fn`` function is used to produce
        configurations to be sent to all clients. These are packaged with the provided parameters and set over to the
        clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): The parameters to be used to initialize the clients for the fit round.
            client_manager (ClientManager): The manager used to sample from the available clients.

        Returns:
            list[tuple[ClientProxy, FitIns]]: List of sampled client identifiers and the configuration/parameters to
            be sent to each client (packaged as ``FitIns``).
        """
        if isinstance(client_manager, BaseFractionSamplingManager):
            # Using one of the custom FractionSamplingManager classes, sampling fraction is based on fraction_fit
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)
            else:
                config = {"current_server_round": server_round}
            fit_ins = FitIns(parameters, config)

            # Sample clients
            clients = client_manager.sample_fraction(self.fraction_fit, self.min_available_clients)

            # Return client/config pairs
            return [(client, fit_ins) for client in clients]
        log(INFO, f"Using the standard Flower ClientManager: {type(client_manager)}")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        This function configures a sample of clients for a evaluation round. It handles the case where the client
        manager has a sample fraction vs. a sample function (to allow for more flexible sampling).
        The function follows the standard configuration flow where the ``on_evaluate_config_fn`` function is used to
        produce configurations to be sent to all clients. These are packaged with the provided parameters and set over
        to the clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): The parameters to be used to initialize the clients for the eval round.
            client_manager (ClientManager): The manager used to sample from the available clients.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of sampled client identifiers and the configuration/parameters
            to be sent to each client (packaged as ``EvaluateIns``).
        """
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
            else:
                config = {"current_server_round": server_round}
            evaluate_ins = EvaluateIns(parameters, config)

            # Sample clients
            clients = client_manager.sample_fraction(self.fraction_evaluate, self.min_available_clients)

            # Return client/config pairs
            return [(client, evaluate_ins) for client in clients]
        log(INFO, f"Using the standard Flower ClientManager: {type(client_manager)}")
        return super().configure_evaluate(server_round, parameters, client_manager)

    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, GetPropertiesIns]]:
        """
        This function configures everything required to request properties from **ALL** of the clients. The client
        manger, regardless of type, is instructed to grab all available clients to perform the polling process.

        Args:
            server_round (int): Indicates the server round we're currently on.
            client_manager (ClientManager): The manager used to sample all available clients.

        Returns:
            list[tuple[ClientProxy, GetPropertiesIns]]: List of sampled client identifiers and the configuration
            to be sent to each client (packaged as ``GetPropertiesIns``).
        """
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
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate the results from the federated fit round. This is done with either weighted or unweighted FedAvg,
        depending on the settings used for the strategy.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (list[tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            tuple[Parameters | None, dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Sorting the results by elements and sample counts. This is primarily to reduce numerical fluctuations in
        # summing the numpy arrays during aggregation. This ensures that addition will occur in the same order,
        # reducing numerical fluctuation.
        decoded_and_sorted_results = [
            (weights, sample_counts) for _, weights, sample_counts in decode_and_pseudo_sort_results(results)
        ]

        # Aggregate them in a weighted or unweighted fashion based on settings.
        aggregated_arrays = aggregate_results(decoded_and_sorted_results, self.weighted_aggregation)
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
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregate the metrics and losses returned from the clients as a result of the evaluation round.

        Args:
            server_round (int): Current FL server Round.
            results (list[tuple[ClientProxy, EvaluateRes]]): The client identifiers and the results of their local
                evaluation that need to be aggregated on the server-side. These results are loss values and the
                metrics dictionary.
            failures (list[tuple[ClientProxy, EvaluateRes]  |  BaseException]): These are the results and
                exceptions from clients that experienced an issue during evaluation, such as timeouts or exceptions.

        Returns:
            tuple[float | None, dict[str, Scalar]]: Aggregated loss values and the aggregated metrics. The metrics
            are aggregated according to ``evaluate_metrics_aggregation_fn``.
        """
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


class OpacusBasicFedAvg(BasicFedAvg):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        model: GradSampleModule,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
        """
        This strategy is a simple extension of the BasicFedAvg strategy to force the model being federally trained to
        be an valid Opacus ``GradSampleModule`` and, thereby, ensure that associated the parameters are aligned with
        those of Opacus based models used by the ``InstanceLevelDpClient``.

        Args:
            model (GradSampleModule): The model architecture to be federally trained. When using this strategy,
                the model must be of type Opacus ``GradSampleModule``. This model will then be used to set
                ``initialize_parameters`` as the initial parameters to be used by all clients.
            fraction_fit (float, optional): Fraction of clients used during training. In case ``min_fit_clients`` is
                larger than ``fraction_fit * available_clients``, ``min_fit_clients`` will still be sampled.
                Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. In case
                ``min_evaluate_clients`` is larger than ``fraction_evaluate * available_clients``,
                ``min_evaluate_clients`` will still be sampled. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                client-side validation by providing a ``Config`` dictionary. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
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
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
        assert isinstance(model, GradSampleModule), "Provided model must be Opacus type GradSampleModule"
        # Setting the initial parameters to correspond with those of the provided model
        self.initial_parameters = get_all_model_parameters(model)
