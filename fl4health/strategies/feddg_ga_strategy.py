from enum import Enum
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import EvaluateRes, FitIns, FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager


class SignalForTypeException(Exception):
    """Thrown when there is an error in `signal_for_type` function."""

    pass


class FairnessMetricType(Enum):
    """Defines the basic types for fairness metrics, their default names and their default signals"""

    ACCURACY = "val - prediction - accuracy"
    LOSS = "val - loss"
    CUSTOM = "custom"

    @classmethod
    def signal_for_type(cls, fairness_metric_type: "FairnessMetricType") -> float:
        """
        Return the default signal for the given metric type.

        Args:
            fairness_metric_type: (FairnessMetricType) the fairness metric type.

        Returns: (float) -1.0 if FairnessMetricType.ACCURACY or 1.0 if FairnessMetricType.LOSS.

        Raises: (SignalForTypeException) if type is CUSTOM as the signal has to be defined by the user.
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
        raise SignalForTypeException("This function should not be called with CUSTOM type.")


class FairnessMetric:
    """Defines a fairness metric with attributes that can be overridden if needed."""

    def __init__(
        self,
        metric_type: FairnessMetricType,
        metric_name: Optional[str] = None,
        signal: Optional[float] = None,
    ):
        """
        Instantiates a fairness metric with a type and optional metric name and
            signal if one wants to override them.

        Args:
            metric_type: (FairnessMetricType) the fairness metric type. If CUSTOM, the metric_name and
                signal should be provided.
            metric_name: (str, optional) the name of the metric to be used as fairness metric.
                Optional, default is metric_type.value. Mandatory if metric_type is CUSTOM.
            signal: (float, optional) the signal of the fairness metric. Optional, default is
                FairnessMetricType.signal_for_type(metric_type). Mandatory if metric_type is CUSTOM.
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


class FedDgGaStrategy(FedAvg):
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
        fairness_metric: Optional[FairnessMetric] = None,
        weight_step_size: float = 0.2,
    ):
        """Strategy for the FedDG-GA algorithm (Federated Domain Generalization with
        Generalization Adjustment, Zhang et al. 2023).

        Args:
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
            evaluate_fn :
                Optional[
                    Callable[[int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]]]
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
            fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            fairness_metric : FairnessMetric, optional.
                The metric to evaluate the local model of each client against the global model in order to
                determine their adjustment weight for aggregation. Can be set to any default metric in
                FairnessMetricType or set to use a custom metric. Optional, default is
                FairnessMetric(FairnessMetricType.LOSS).
            weight_step_size : float
                The step size to determine the magnitude of change for the adjustment weight. It has to be
                0 < weight_step_size < 1. Optional, default is 0.2.
        """
        if fraction_fit != 1.0 or fraction_evaluate != 1.0:
            log(
                WARNING,
                "fraction_fit or fraction_evaluate are not 1.0. The behaviour of FedDG-GA is unknown in those cases.",
            )

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

        if fairness_metric is None:
            self.fairness_metric = FairnessMetric(FairnessMetricType.LOSS)
        else:
            self.fairness_metric = fairness_metric

        self.weight_step_size = weight_step_size
        assert 0 < self.weight_step_size < 1, f"weight_step_size has to be between 0 and 1 ({self.weight_step_size})"

        self.train_metrics: Dict[str, Dict[str, Scalar]] = {}
        self.evaluation_metrics: Dict[str, Dict[str, Scalar]] = {}
        self.num_rounds: Optional[int] = None
        self.initial_adjustment_weight: Optional[float] = None
        self.adjustment_weights: Dict[str, float] = {}

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training.

        Will also collect the number of rounds the training will run for in order
        to calculate the adjustment weight step size. Fails if n_server_rounds is not set in the config
        or if it's not an integer.

        Args:
            server_round: (int) the current server round.
            parameters: (Parameters) the model parameters.
            client_manager: (ClientManager) The client manager which holds all currently
                connected clients. It must be an instance of FixedSamplingClientManager.

        Returns:
            (List[Tuple[ClientProxy, FitIns]]) the input for the clients' fit function.
        """
        assert isinstance(
            client_manager, FixedSamplingClientManager
        ), f"Client manager is not of type FixedSamplingClientManager: {type(client_manager)}"

        client_manager.reset_sample()

        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)

        self.initial_adjustment_weight = 1.0 / len(client_fit_ins)

        # Setting self.num_rounds
        if self.num_rounds is None:
            assert self.on_fit_config_fn is not None, "on_fit_config_fn must be specified"
            config = self.on_fit_config_fn(server_round)
            assert "evaluate_after_fit" in config, "evaluate_after_fit must be present in config and set to True"
            assert config["evaluate_after_fit"] is True, "evaluate_after_fit must be set to True"
            assert "n_server_rounds" in config, "n_server_rounds must be specified"
            assert isinstance(config["n_server_rounds"], int), "n_server_rounds is not an integer"
            self.num_rounds = config["n_server_rounds"]

        return client_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results by weighing them against the adjustment weights and then summing them.

        Collects the fit metrics that will be used to change the adjustment weights for the next round.

        Args:
            server_round: (int) the current server round.
            results: (List[Tuple[ClientProxy, FitRes]]) The clients' fit results.
            failures: (List[Union[Tuple[ClientProxy, FitRes], BaseException]]) the clients' fit failures.

        Returns:
            (Tuple[Optional[Parameters], Dict[str, Scalar]]) A tuple containing the aggregated parameters
                and the aggregated fit metrics.
        """
        # The original aggregated parameters is done by the super class (which we want to
        # override its behaviour here), so we are discarding it to recalculate them in the lines below
        _, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        self.train_metrics = {}
        for client_proxy, fit_res in results:
            self.train_metrics[client_proxy.cid] = fit_res.metrics

        parameters_aggregated = ndarrays_to_parameters(self.weight_and_aggregate_results(results))

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation losses using weighted average.

        Collects the evaluation metrics and updates the adjustment weights, which will be used
        when aggregating the results for the next round.

        Args:
            server_round: (int) the current server round.
            results: (List[Tuple[ClientProxy, FitRes]]) The clients' evaluate results.
            failures: (List[Union[Tuple[ClientProxy, FitRes], BaseException]]) the clients' evaluate failures.

        Returns:
            (Tuple[Optional[float], Dict[str, Scalar]]) A tuple containing the aggregated evaluation loss
                and the aggregated evaluation metrics.
        """
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.evaluation_metrics = {}
        for client_proxy, eval_res in results:
            cid = client_proxy.cid
            self.evaluation_metrics[cid] = eval_res.metrics
            # adding the loss to the metrics
            val_loss_key = FairnessMetricType.LOSS.value
            self.evaluation_metrics[cid][val_loss_key] = eval_res.loss

        # Updating the weights at the end of the training round
        cids = [client_proxy.cid for client_proxy, _ in results]
        self.update_weights_by_ga(server_round, cids)

        return loss_aggregated, metrics_aggregated

    def weight_and_aggregate_results(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        """
        Aggregate results by weighing them against the adjustment weights and then summing them.

        Args:
            results: (List[Tuple[ClientProxy, FitRes]]) The clients' fit results.

        Returns:
            (NDArrays) the weighted and aggregated results.
        """

        aggregated_results: Optional[NDArrays] = None
        for client_proxy, fit_res in results:
            cid = client_proxy.cid

            # initializing adjustment weights for this client if they don't exist yet
            if cid not in self.adjustment_weights:
                assert self.initial_adjustment_weight is not None
                self.adjustment_weights[cid] = self.initial_adjustment_weight

            # apply adjustment weights
            weighted_client_parameters = parameters_to_ndarrays(fit_res.parameters)
            for i in range(len(weighted_client_parameters)):
                weighted_client_parameters[i] = weighted_client_parameters[i] * self.adjustment_weights[cid]

            # sum weighted parameters
            if aggregated_results is None:
                aggregated_results = weighted_client_parameters
            else:
                assert len(weighted_client_parameters) == len(aggregated_results)
                for i in range(len(weighted_client_parameters)):
                    aggregated_results[i] = aggregated_results[i] + weighted_client_parameters[i]

        assert aggregated_results is not None
        return aggregated_results

    def update_weights_by_ga(self, server_round: int, cids: List[str]) -> None:
        """
        Update the self.adjustment_weights dictionary by calculating the new weights
        based on the current server round, fit and evaluation metrics.

        Args:
            server_round: (int) the current server round.
            cids: (List[str]) the list of client ids that participated in this round.
        """
        generalization_gaps = []
        # calculating local vs global metric difference (generalization gaps)
        for cid in cids:
            assert (
                cid in self.train_metrics and cid in self.evaluation_metrics
            ), f"{cid} not in {self.train_metrics.keys()} or {self.evaluation_metrics.keys()}"

            assert self.fairness_metric.metric_name is not None

            global_model_metric_value = self.evaluation_metrics[cid][self.fairness_metric.metric_name]
            local_model_metric_value = self.train_metrics[cid][self.fairness_metric.metric_name]
            assert isinstance(global_model_metric_value, float) and isinstance(local_model_metric_value, float)

            generalization_gaps.append(global_model_metric_value - local_model_metric_value)

        # Calculating the normalized generalization gaps
        generalization_gaps_ndarray = np.array(generalization_gaps)
        mean_generalization_gap = np.mean(generalization_gaps_ndarray)
        var_generalization_gaps = generalization_gaps_ndarray - mean_generalization_gap
        max_var_generalization_gap = np.max(np.abs(var_generalization_gaps))

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

    def get_current_weight_step_size(self, server_round: int) -> float:
        """
        Calculates the current weight step size based on the current server round,  weight
        step size and total number of rounds.

        Args:
            server_round: (int) the current server round

        Returns: (float) the current value for the weight step size.
        """
        # The implementation of d^r here differs from the definition in the paper
        # because our server round starts at 1 instead of 0.
        assert self.num_rounds is not None
        weight_step_size_decay = self.weight_step_size / self.num_rounds
        weight_step_size_for_round = self.weight_step_size - ((server_round - 1) * weight_step_size_decay)

        # Omitting an additional scaler here that is present in the reference
        # implementation but not in the paper:
        # weight_step_size_for_round *= self.initial_adjustment_weight

        return weight_step_size_for_round
