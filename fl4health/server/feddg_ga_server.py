from enum import Enum
from logging import ERROR
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServer


class FixedSamplingClientManager(SimpleClientManager):
    """Keeps sampling fixed until it's reset"""

    def __init__(self) -> None:
        super().__init__()
        self.current_sample: Optional[List[ClientProxy]] = None

    def reset_sample(self) -> None:
        """Resets the saved sample so self.sample produces a new sample again."""
        self.current_sample = None

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """
        Return a new client sample for the first time it runs. For subsequent runs,
        it will return the sample sampling until self.reset_sampling() is called.

        Args:
            num_clients: (int) The number of clients to sample.
            min_num_clients: (Optional[int]) The minimum number of clients to return in the sample.
                Optional, default is num_clients.
            criterion: (Optional[Criterion]) A criterion to filter clients to sample.
                Optional, default is no criterion (no filter).

        Returns:
            List[ClientProxy]: A list of sampled clients as ClientProxy instances.
        """
        if self.current_sample is None:
            self.current_sample = super().sample(num_clients, min_num_clients, criterion)
        return self.current_sample


class ClientMetrics:
    """Stores client metrics for easy retrieval."""

    def __init__(
        self,
        train_metrics: Optional[Dict[str, Scalar]] = None,
        evaluation_metrics: Optional[Dict[str, Scalar]] = None,
    ):
        """
        Args:
            train_metrics: (Optional[Dict[str, Scalar]]) a list of train metrics. Optional, default is None.
            evaluation_metrics: (Optional[Dict[str, Scalar]]) a list of evaluation metrics. Optional, default is None.
        """
        self.train_metrics = train_metrics
        self.evaluation_metrics = evaluation_metrics

    def __repr__(self) -> str:
        return f"train_metrics: {self.train_metrics}, evaluation_metrics: {self.evaluation_metrics}"


class FairnessMetricType(Enum):
    # TODO docstrings
    ACCURACY = "val - prediction - accuracy"
    LOSS = "val - loss"
    CUSTOM = "custom"

    @classmethod
    def signal_for_type(cls, fairness_metric_type: "FairnessMetricType") -> float:
        if fairness_metric_type == FairnessMetricType.ACCURACY:
            return -1.0
        if fairness_metric_type == FairnessMetricType.LOSS:
            return 1
        return 0


class FairnessMetric:
    # TODO docstrings
    def __init__(
        self,
        metric_type: FairnessMetricType,
        metric_name: str = "",
        signal: float = 0.0,
    ):
        # TODO dosctrings
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.signal = signal

        if metric_type is None:
            assert metric_name is not None and signal is not None
        else:
            if metric_name == "":
                self.metric_name = metric_type.value
            if signal == 0.0:
                self.signal = FairnessMetricType.signal_for_type(metric_type)


class FedDGGAServer(FlServer):
    def __init__(
        self,
        client_manager: FixedSamplingClientManager,
        fairness_metric: Optional[FairnessMetric] = None,
        weight_step_size: float = 0.2,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        # TODO docstrings
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer, metrics_reporter)

        if fairness_metric is None:
            self.fairness_metric = FairnessMetric(FairnessMetricType.LOSS)
        else:
            self.fairness_metric = fairness_metric

        self.weight_step_size = weight_step_size
        self.clients_metrics: List[ClientMetrics] = []
        self.num_rounds: Optional[int] = None
        self.adjustment_weights: Dict[str, float] = {}
        self.results_and_failures: FitResultsAndFailures = ([], [])

    def client_manager(self) -> FixedSamplingClientManager:
        """Return FixedSamplingClientManager."""
        client_manager = super().client_manager()
        assert isinstance(client_manager, FixedSamplingClientManager), (
            "Client manager is not of type FixedSamplingClientManager" f"({type(client_manager)})"
        )
        return client_manager

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        # TODO docstring
        # Storing the number of rounds to be used for calculating the weight step decay
        self.num_rounds = num_rounds
        return super().fit(num_rounds, timeout)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        # TODO docstrings

        # Resetting sampling and metrics
        self.client_manager().reset_sample()
        self.clients_metrics = []

        res_fit = super().fit_round(server_round, timeout)

        # Collecting train metrics
        # should contain evaluation of the client's local model against its validation set
        if res_fit:
            # TODO what to do in case of failure?
            _, _, (results, failures) = res_fit
            self.results_and_failures = (results, failures)
            for result in results:
                self.clients_metrics.append(ClientMetrics(train_metrics=result[1].metrics))

        return res_fit

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        # TODO docstrings
        res_eval = super().evaluate_round(server_round, timeout)

        # Collecting evaluation metrics
        # which consists of evaluation of the global model against the client's validation set
        if res_eval:
            # TODO what to do in case of failure?
            loss_fed, _, (results, _) = res_eval
            for i in range(len(results)):
                self.clients_metrics[i].evaluation_metrics = results[i][1].metrics
                # adding the loss to the metrics
                val_loss_key = FairnessMetricType.LOSS.value
                self.clients_metrics[i].evaluation_metrics[val_loss_key] = loss_fed  # type: ignore

        # Calculating and applying the weights at the end of the training round
        self.calculate_weights_by_ga(server_round)
        self.apply_weights_to_results()

        # Making the aggregation again, now with the weighted results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, self.results_and_failures[0], self.results_and_failures[1])
        parameters_aggregated, _ = aggregated_result

        if parameters_aggregated:
            self.parameters = parameters_aggregated
        else:
            log(ERROR, "Parameters aggregated is None.")

        return res_eval

    def calculate_weights_by_ga(self, server_round: int) -> None:
        # TODO docstrings
        clients_proxies = self.client_manager().current_sample
        assert clients_proxies is not None

        value_list = []
        # calculating local vs global metric difference
        for i in range(len(self.clients_metrics)):
            client_metrics = self.clients_metrics[i]
            assert client_metrics.evaluation_metrics is not None and client_metrics.train_metrics is not None

            global_model_metric_value = client_metrics.evaluation_metrics[self.fairness_metric.metric_name]
            local_model_metric_value = client_metrics.train_metrics[self.fairness_metric.metric_name]
            assert isinstance(global_model_metric_value, float) and isinstance(local_model_metric_value, float)

            value_list.append(global_model_metric_value - local_model_metric_value)

            # initializing weight for this client
            cid = clients_proxies[i].cid
            if cid not in self.adjustment_weights:
                self.adjustment_weights[cid] = 1.0 / 3.0

        # calculating norm gap
        value_list_ndarray = np.array(value_list)
        norm_gap_list = value_list_ndarray / np.max(np.abs(value_list_ndarray))
        step_size = 1.0 / 3.0 * self.get_current_weight_step_size(server_round)

        # updating weights
        for i in range(len(self.clients_metrics)):
            cid = clients_proxies[i].cid
            self.adjustment_weights[cid] += self.fairness_metric.signal * norm_gap_list[i] * step_size

        # weight clip
        new_total_weight = 0.0
        for cid in self.adjustment_weights:
            clipped_weight = np.clip(self.adjustment_weights[cid], 0.0, 1.0)
            self.adjustment_weights[cid] = clipped_weight
            new_total_weight += clipped_weight

        for cid in self.adjustment_weights:
            self.adjustment_weights[cid] /= new_total_weight

    def get_current_weight_step_size(self, server_round: int) -> float:
        # TODO docstring
        assert self.num_rounds is not None
        weight_step_size_decay = self.weight_step_size / self.num_rounds
        weight_step_size_for_round = self.weight_step_size - ((server_round - 1) * weight_step_size_decay)
        return weight_step_size_for_round

    def apply_weights_to_results(self) -> None:
        # TODO docstring
        clients_proxies = self.client_manager().current_sample
        assert clients_proxies is not None

        for i in range(len(self.results_and_failures[0])):
            cid = clients_proxies[i].cid
            client_weight = self.adjustment_weights[cid]
            client_results = self.results_and_failures[0][i]
            client_parameters = parameters_to_ndarrays(client_results[1].parameters)

            for j in range(len(client_parameters)):
                client_parameters[i] *= client_weight

            weighted_client_parameters = ndarrays_to_parameters(client_parameters)
            client_results[1].parameters = weighted_client_parameters
