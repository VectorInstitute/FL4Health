from collections.abc import Callable
from enum import Enum
from logging import INFO, WARNING

import numpy as np
from flwr.common import EvaluateIns, MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import EvaluateRes, FitIns, FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.utils.functions import decode_and_pseudo_sort_results


class SignalForTypeError(Exception):
    """Thrown when there is an error in ``signal_for_type`` function."""

    pass


class FairnessMetricType(Enum):
    """Defines the basic types for fairness metrics, their default names and their default signals."""

    ACCURACY = "val - prediction - accuracy"
    LOSS = "val - checkpoint"
    CUSTOM = "custom"

    @classmethod
    def signal_for_type(cls, fairness_metric_type: "FairnessMetricType") -> float:
        """
        Return the default signal for the given metric type.

        Args:
            fairness_metric_type (FairnessMetricType): the fairness metric type.

        Raises:
            SignalForTypeException: if type is ``CUSTOM`` as the signal has to be defined by the user.

        Returns:
            (float): -1.0 if ``FairnessMetricType.ACCURACY`` or 1.0 if ``FairnessMetricType.LOSS``.
        """
        # For loss values, large and **positive** gaps imply worse generalization of global
        # weights to local models. Therefore, we want to **increase** weight for these model
        # parameters to improve generalization. So signal is positive. For accuracy, large
        # **negative** gaps imply worse generalization. So the signal is -1.0, to increase
        # weights for the associated model parameters.
        if fairness_metric_type == FairnessMetricType.ACCURACY:
            return -1.0
        if fairness_metric_type == FairnessMetricType.LOSS:
            return 1.0
        raise SignalForTypeError("This function should not be called with CUSTOM type.")


class FairnessMetric:
    def __init__(
        self,
        metric_type: FairnessMetricType,
        metric_name: str | None = None,
        signal: float | None = None,
    ):
        """
        Defines a fairness metric with attributes that can be overridden if needed.

        Instantiates a fairness metric with a type and optional metric name and signal if one wants to override them.

        Args:
            metric_type (FairnessMetricType): the fairness metric type. If ``CUSTOM``, the ``metric_name`` and
                signal should be provided.
            metric_name (str | None, optional): the name of the metric to be used as fairness metric.
                Mandatory if ``metric_type`` is ``CUSTOM``. Defaults to None.
            signal (float | None, optional): the signal of the fairness metric.
                Mandatory if ``metric_type`` is ``CUSTOM``. Defaults to None.
        """
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.signal = signal

        if metric_type is FairnessMetricType.CUSTOM:
            assert metric_name is not None and signal is not None
        else:
            if metric_name is None:
                self.metric_name = metric_type.value
            if signal is None:
                self.signal = FairnessMetricType.signal_for_type(metric_type)

    def __str__(self) -> str:
        """
        String produced when calling str(...) on a Fairness metric object.

        Returns:
            (str): Custom string describing the object attributes.
        """
        return f"Metric Type: {self.metric_type}, Metric Name: '{self.metric_name}', Signal: {self.signal}"


class FedDgGa(FedAvg):
    def __init__(
        self,
        *,
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
        fairness_metric: FairnessMetric | None = None,
        adjustment_weight_step_size: float = 0.2,
    ):
        """
        Strategy for the FedDG-GA algorithm (Federated Domain Generalization with Generalization Adjustment, Zhang et
        al. 2023). This strategy assumes (and checks) that the configuration sent by the server to the clients has the
        key "evaluate_after_fit" and it is set to True. It also ensures that the key "pack_losses_with_val_metrics" is
        present and its value is set to True. These are to facilitate the exchange of evaluation information needed
        for the strategy to work correctly.

        **NOTE**: For FedDG-GA, we require that ``fraction_fit`` and ``fraction_evaluate`` are 1.0, as behavior of the
        FedDG-GA algorithm is not well-defined when participation in each round of training and evaluation is partial.
        Thus, we force these values to be 1.0 in super and do not allow them to be set by the user.

        Args:
            min_fit_clients (int, optional): Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None):
                Optional function used for validation.. Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training. Must be specified for this strategy.. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                validation. Must be specified for this strategy.. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Parameters | None, optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            fairness_metric (FairnessMetric | None, optional): The metric to evaluate the local model of each client
                against the global model in order to determine their adjustment weight for aggregation. Can be set to
                any default metric in ``FairnessMetricType`` or set to use a custom metric. Defaults to None.
            adjustment_weight_step_size (float, optional): The step size to determine the magnitude of change for the
                generalization adjustment weights. It has to be ``0 < adjustment_weight_step_size < 1``.
                Defaults to 0.2.
        """
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
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

        if fairness_metric is None:
            self.fairness_metric = FairnessMetric(FairnessMetricType.LOSS)
        else:
            self.fairness_metric = fairness_metric

        self.adjustment_weight_step_size = adjustment_weight_step_size
        assert 0 < self.adjustment_weight_step_size < 1, (
            f"adjustment_weight_step_size has to be between 0 and 1 ({self.adjustment_weight_step_size})"
        )

        log(INFO, f"FedDG-GA Strategy initialized with weight_step_size of {self.adjustment_weight_step_size}")
        log(INFO, f"FedDG-GA Strategy initialized with FairnessMetric {self.fairness_metric}")

        self.train_metrics: dict[str, dict[str, Scalar]] = {}
        self.evaluation_metrics: dict[str, dict[str, Scalar]] = {}
        self.num_rounds: int | None = None
        self.initial_adjustment_weight: float | None = None
        self.adjustment_weights: dict[str, float] = {}

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training. Will also collect the number of rounds the training will run for in order
        to calculate the adjustment weight step size. Fails if ``n_server_rounds`` is not set in the config or if it's
        not an integer.

        Args:
            server_round (int): The current server round.
            parameters (Parameters): The model parameters.
            client_manager (ClientManager): The client manager which holds all currently connected clients. It must
                be an instance of ``FixedSamplingClientManager``.

        Returns:
            (list[tuple[ClientProxy, FitIns]]): the input for the clients' fit function.
        """
        assert isinstance(client_manager, FixedSamplingClientManager), (
            f"Client manager is not of type FixedSamplingClientManager: {type(client_manager)}"
        )

        client_manager.reset_sample()

        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)

        self.initial_adjustment_weight = 1.0 / len(client_fit_ins)

        # Setting self.num_rounds once and doing some sanity checks
        assert self.on_fit_config_fn is not None, "on_fit_config_fn must be specified"
        config = self.on_fit_config_fn(server_round)
        assert "evaluate_after_fit" in config, "evaluate_after_fit must be present in config"
        assert config["evaluate_after_fit"] is True, "evaluate_after_fit must be set to True"

        assert "pack_losses_with_val_metrics" in config, "pack_losses_with_val_metrics must be present in config"
        assert config["pack_losses_with_val_metrics"] is True, "pack_losses_with_val_metrics must be set to True"

        assert "n_server_rounds" in config, "n_server_rounds must be specified"
        assert isinstance(config["n_server_rounds"], int), "n_server_rounds is not an integer"
        n_server_rounds = config["n_server_rounds"]

        if self.num_rounds is None:
            self.num_rounds = n_server_rounds
        else:
            assert n_server_rounds == self.num_rounds, (
                f"n_server_rounds has changed from the original value of {self.num_rounds} "
                f"and is now {n_server_rounds}"
            )

        return client_fit_ins

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        assert isinstance(client_manager, FixedSamplingClientManager), (
            f"Client manager is not of type FixedSamplingClientManager: {type(client_manager)}"
        )

        client_evaluate_ins = super().configure_evaluate(server_round, parameters, client_manager)

        assert self.on_evaluate_config_fn is not None, "on_fit_config_fn must be specified"
        config = self.on_evaluate_config_fn(server_round)
        assert "pack_losses_with_val_metrics" in config, "pack_losses_with_val_metrics must be present in config"
        assert config["pack_losses_with_val_metrics"] is True, "pack_losses_with_val_metrics must be set to True"

        return client_evaluate_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate fit results by weighing them against the adjustment weights and then summing them.

        Collects the fit metrics that will be used to change the adjustment weights for the next round.

        Args:
            server_round (int): The current server round.
            results (list[tuple[ClientProxy, FitRes]]): The clients' fit results.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): The clients' fit failures.

        Returns:
            (tuple[Parameters | None, dict[str, Scalar]]): A tuple containing the aggregated parameters and the
                aggregated fit metrics.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.train_metrics = {}
        for client_proxy, fit_res in results:
            self.train_metrics[client_proxy.cid] = fit_res.metrics

        parameters_aggregated = ndarrays_to_parameters(self.weight_and_aggregate_results(results))

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregate evaluation losses using weighted average.

        Collects the evaluation metrics and updates the adjustment weights, which will be used when aggregating the
        results for the next round.

        Args:
            server_round (int): The current server round.
            results (list[tuple[ClientProxy, FitRes]]): The clients' evaluate results.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): the clients' evaluate failures.

        Returns:
            (tuple[float | None, dict[str, Scalar]]): A tuple containing the aggregated evaluation loss and the
                aggregated evaluation metrics.
        """
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.evaluation_metrics = {}
        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            # make sure that the metrics has the desired loss key
            assert FairnessMetricType.LOSS.value in eval_res.metrics
            self.evaluation_metrics[cid] = eval_res.metrics

        # Updating the weights at the end of the training round
        cids = [client_proxy.cid for client_proxy, _ in results]
        log(INFO, "Updating the Generalization Adjustment Weights")
        self.update_weights_by_ga(server_round, cids)

        return loss_aggregated, metrics_aggregated

    def weight_and_aggregate_results(self, results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
        """
        Aggregate results by weighing them against the adjustment weights and then summing them.

        Args:
            results (list[tuple[ClientProxy, FitRes]]): The clients' fit results.

        Returns:
            (NDArrays): The weighted and aggregated results.
        """
        if self.adjustment_weights:
            log(INFO, f"Current adjustment weights by Client ID (CID) are {self.adjustment_weights}")
        else:
            # If the adjustment weights dictionary doesn't exist, it means that it hasn't been initialized
            # and will be below.
            log(INFO, f"Current adjustment weights are all initialized to {self.initial_adjustment_weight}")

        # Sorting the results by elements and sample counts. This is primarily to reduce numerical fluctuations in
        # summing the numpy arrays during aggregation. This ensures that addition will occur in the same order,
        # reducing numerical fluctuation.
        decoded_and_sorted_results = decode_and_pseudo_sort_results(results)

        aggregated_results: NDArrays | None = None
        for client_proxy, weights, _ in decoded_and_sorted_results:
            cid = client_proxy.cid

            # initializing adjustment weights for this client if they don't exist yet
            if cid not in self.adjustment_weights:
                assert self.initial_adjustment_weight is not None
                self.adjustment_weights[cid] = self.initial_adjustment_weight

            # apply adjustment weights
            weighted_client_parameters = weights
            for i in range(len(weighted_client_parameters)):
                weighted_client_parameters[i] = weighted_client_parameters[i] * self.adjustment_weights[cid]

            # sum weighted parameters
            if aggregated_results is None:
                # If this is the first client we're applying adjustment to, we set the results to those parameters.
                # Remaining client parameters will be subsequently added to these.
                aggregated_results = weighted_client_parameters
            else:
                assert len(weighted_client_parameters) == len(aggregated_results)
                for i in range(len(weighted_client_parameters)):
                    aggregated_results[i] = aggregated_results[i] + weighted_client_parameters[i]

        assert aggregated_results is not None
        return aggregated_results

    def update_weights_by_ga(self, server_round: int, cids: list[str]) -> None:
        """
        Update the ``self.adjustment_weights`` dictionary by calculating the new weights based on the current server
        round, fit and evaluation metrics.

        Args:
            server_round (int): The current server round.
            cids (list[str]): The list of client ids that participated in this round.
        """
        generalization_gaps = []
        # calculating local vs global metric difference (generalization gaps)
        for cid in cids:
            assert cid in self.train_metrics and cid in self.evaluation_metrics, (
                f"{cid} not in {self.train_metrics.keys()} or {self.evaluation_metrics.keys()}"
            )

            assert self.fairness_metric.metric_name is not None

            global_model_metric_value = self.evaluation_metrics[cid][self.fairness_metric.metric_name]
            local_model_metric_value = self.train_metrics[cid][self.fairness_metric.metric_name]
            assert isinstance(global_model_metric_value, float) and isinstance(local_model_metric_value, float)

            generalization_gaps.append(global_model_metric_value - local_model_metric_value)

        log(
            INFO,
            "Client ID (CID) and Generalization Gaps (G_{{hat{{D_i}}}}(theta^r)): "
            f"{list(zip(cids, generalization_gaps))}",
        )

        # Calculating the normalized generalization gaps
        generalization_gaps_ndarray = np.array(generalization_gaps)
        mean_generalization_gap = np.mean(generalization_gaps_ndarray)
        var_generalization_gaps = generalization_gaps_ndarray - mean_generalization_gap
        max_var_generalization_gap = np.max(np.abs(var_generalization_gaps))
        log(INFO, f"Mean Generalization Gap (mu): {mean_generalization_gap}")
        log(INFO, f"Max Absolute Deviation of Generalization Gaps: {max_var_generalization_gap}")

        if max_var_generalization_gap == 0:
            log(
                WARNING,
                "Max variance in generalization gap is 0. Adjustment weights will remain the same. "
                + f"Generalization gaps: {generalization_gaps}",
            )
            normalized_generalization_gaps = np.zeros_like(generalization_gaps)
        else:
            step_size = self.get_current_weight_step_size(server_round)
            normalized_generalization_gaps = (var_generalization_gaps * step_size) / max_var_generalization_gap

        # updating weights
        new_total_weight = 0.0
        for i in range(len(cids)):
            cid = cids[i]
            # For loss values, large and **positive** gaps imply worse generalization of global
            # weights to local models. Therefore, we want to **increase** weight for these model
            # parameters to improve generalization. So signal is positive. For accuracy, large
            # **negative** gaps imply worse generalization. So the signal is -1.0, to increase
            # weights for the associated model parameters.
            self.adjustment_weights[cid] += self.fairness_metric.signal * normalized_generalization_gaps[i]

            # Weight clip
            # The paper states the clipping only happens for values below 0 but the reference
            # implementation also clips values larger than 1, probably as an extra assurance.
            clipped_weight = np.clip(self.adjustment_weights[cid], 0.0, 1.0)
            self.adjustment_weights[cid] = clipped_weight
            new_total_weight += clipped_weight

        for cid in cids:
            self.adjustment_weights[cid] /= new_total_weight
        log(INFO, f"New Generalization Adjustment Weights by Client ID (CID) are {self.adjustment_weights}")

    def get_current_weight_step_size(self, server_round: int) -> float:
        """
        Calculates the current weight step size based on the current server round,  weight step size and total number
        of rounds.

        Args:
            server_round (int): the current server round

        Returns (float):
            the current value for the weight step size.
        """
        # The implementation of d^r here differs from the definition in the paper
        # because our server round starts at 1 instead of 0.
        assert self.num_rounds is not None
        weight_step_size_decay = self.adjustment_weight_step_size / self.num_rounds
        weight_step_size_for_round = self.adjustment_weight_step_size - ((server_round - 1) * weight_step_size_decay)
        log(
            INFO, f"Step size for round: {weight_step_size_for_round}, original was {self.adjustment_weight_step_size}"
        )

        # Omitting an additional scaler here that is present in the reference
        # implementation but not in the paper:
        # weight_step_size_for_round *= self.initial_adjustment_weight

        return weight_step_size_for_round
