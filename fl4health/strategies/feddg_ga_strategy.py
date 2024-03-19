from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import EvaluateRes, FitIns, FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager


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

        Returns: (float) -1.0 if FairnessMetricType.ACCURACY, 1.0 if FairnessMetricType.LOSS and
            0 if FairnessMetricType.CUSTOM.
        """
        if fairness_metric_type == FairnessMetricType.ACCURACY:
            return -1.0
        if fairness_metric_type == FairnessMetricType.LOSS:
            return 1
        return 0


class FairnessMetric:
    """Defines a fairness metric with attributes that can be overridden if needed."""

    def __init__(
        self,
        metric_type: FairnessMetricType,
        metric_name: str = "",
        signal: float = 0.0,
    ):
        """
        Instantiates a fairness metric with a type and optional metric name and
            signal if one wants to override them.

        Args:
            metric_type: (FairnessMetricType) the fairness metric type. If CUSTOM, the metric_type and
                signal should be provided.
            metric_name: (str, optional) the name of the metric to be used as fairness metric.
                Optional, default is metric_type.value.
            signal: (float, optional) the signal of the fairness metric. Optional default is
                FairnessMetricType.signal_for_type(metric_type)
        """
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.signal = signal

        if metric_type is FairnessMetricType.CUSTOM:
            assert metric_name is not None and metric_name != "" and signal is not None
        else:
            if metric_name == "":
                self.metric_name = metric_type.value
            if signal == 0.0:
                self.signal = FairnessMetricType.signal_for_type(metric_type)


class FedDGGAStrategy(FedAvg):
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
        # TODO docstrings

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

        self.train_metrics: Dict[str, Dict[str, Scalar]] = {}
        self.evaluation_metrics: Dict[str, Dict[str, Scalar]] = {}
        self.num_rounds: Optional[int] = None
        self.adjustment_weights: Dict[str, float] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        Parameters
        ----------
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """

        # TODO add this assert to docstrings
        assert isinstance(
            client_manager, FixedSamplingClientManager
        ), f"Client manager is not of type FixedSamplingClientManager: {type(client_manager)}"

        return super().initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # TODO docstrings
        client_fit_ins = super().configure_fit(server_round, parameters, client_manager)

        # Setting self.num_rounds
        if self.num_rounds is None:
            assert self.on_fit_config_fn is not None, "on_fit_config_in must be specified"
            config = self.on_fit_config_fn(server_round)
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
        # TODO docstrings

        # TODO should we call super here or just recalculate the weights?
        _, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        self.train_metrics = {}
        for result in results:
            self.train_metrics[result[0].cid] = result[1].metrics

        parameters_aggregated = ndarrays_to_parameters(self.weight_and_aggregate_results(results))

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # TODO docstrings

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.evaluation_metrics = {}
        for result in results:
            cid = result[0].cid
            self.evaluation_metrics[cid] = result[1].metrics
            # adding the loss to the metrics
            val_loss_key = FairnessMetricType.LOSS.value
            self.evaluation_metrics[cid][val_loss_key] = loss_aggregated  # type: ignore

        # Updating the weights at the end of the training round
        cids = [result[0].cid for result in results]
        self.update_weights_by_ga(server_round, cids)

        return loss_aggregated, metrics_aggregated

    def weight_and_aggregate_results(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        # TODO docstrings

        aggregated_results: Optional[NDArrays] = None
        for result in results:
            cid = result[0].cid

            # initializing adjustment weights for this client if they don't exist yet
            if cid not in self.adjustment_weights:
                self.adjustment_weights[cid] = 1.0 / 3.0

            # apply adjustment weights
            weighted_client_parameters = parameters_to_ndarrays(result[1].parameters)
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
        """
        # TODO add cids to docstrings

        value_list = []
        # calculating local vs global metric difference
        for cid in cids:
            assert cid in self.train_metrics and cid in self.evaluation_metrics

            global_model_metric_value = self.evaluation_metrics[cid][self.fairness_metric.metric_name]
            local_model_metric_value = self.train_metrics[cid][self.fairness_metric.metric_name]
            assert isinstance(global_model_metric_value, float) and isinstance(local_model_metric_value, float)

            value_list.append(global_model_metric_value - local_model_metric_value)

        # calculating norm gap
        value_list_ndarray = np.array(value_list)
        norm_gap_list = value_list_ndarray / np.max(np.abs(value_list_ndarray))
        step_size = 1.0 / 3.0 * self.get_current_weight_step_size(server_round)

        # updating weights
        new_total_weight = 0.0
        for i in range(len(cids)):
            cid = cids[i]
            self.adjustment_weights[cid] += self.fairness_metric.signal * norm_gap_list[i] * step_size

            # weight clip
            clipped_weight = np.clip(self.adjustment_weights[cid], 0.0, 1.0)
            self.adjustment_weights[cid] = clipped_weight
            # TODO Question: should we sum all the weights or just the current clients' weights?
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
        assert self.num_rounds is not None
        weight_step_size_decay = self.weight_step_size / self.num_rounds
        weight_step_size_for_round = self.weight_step_size - ((server_round - 1) * weight_step_size_decay)
        return weight_step_size_for_round
