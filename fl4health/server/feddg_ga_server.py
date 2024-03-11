from enum import Enum
from logging import ERROR
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.fixed_sampling_client_manager import FixedSamplingClientManager
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServer


class ClientMetrics:
    """Stores client metrics for easy retrieval."""

    def __init__(
        self,
        cid: str,
        train_metrics: Optional[Dict[str, Scalar]] = None,
        evaluation_metrics: Optional[Dict[str, Scalar]] = None,
    ):
        """
        Args:
            cid: (str) The client id that generated these metrics
            train_metrics: (Optional[Dict[str, Scalar]]) a list of train metrics. Optional, default is None.
            evaluation_metrics: (Optional[Dict[str, Scalar]]) a list of evaluation metrics. Optional, default is None.
        """
        self.cid = cid
        self.train_metrics = train_metrics
        self.evaluation_metrics = evaluation_metrics

    def __repr__(self) -> str:
        """
        Makes a string representation of this object.

        Returns: (str) a string representation of this object.
        """
        return f"cid: {self.cid}, train_metrics: {self.train_metrics}, evaluation_metrics: {self.evaluation_metrics}"


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


class FedDGGAServer(FlServer):
    """
    A server that implements FedDG-GA (Federated Domain Generalization with Generalization Adjustment,
    Zhang et al. 2023).
    """

    def __init__(
        self,
        fairness_metric: Optional[FairnessMetric] = None,
        weight_step_size: float = 0.2,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        """
        Inits an instance of FedDGGAServer

        Must have "evaluate_after_fit: True" in the config in order to work properly. Will set client_manager
        to a new instance of FixedSamplingClientManager.

        Args:
            fairness_metric: (FairnessMetric, optional) The fairness metric to evaluate the clients against.
                Optional, default is FairnessMetric(FairnessMetricType.LOSS).
            weight_step_size: (float, optional) the step size the GA weights should be adjusted at the end of
                each round. Optional, default is 0.2.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
        """
        super().__init__(FixedSamplingClientManager(), strategy, wandb_reporter, checkpointer, metrics_reporter)

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
        assert isinstance(
            client_manager, FixedSamplingClientManager
        ), f"Client manager is not of type FixedSamplingClientManager: {type(client_manager)}"
        return client_manager

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """
        Run federated learning for a number of rounds.

        Args:
            num_rounds: (int) the number of rounds to run federated learning for.
            timeout: (Optional[float]) the timeout to wait for clients to respond.

        Returns: (History) A History object with the results of training and evaluation.
        """
        # Storing the number of rounds to be used for calculating the weight step decay
        self.num_rounds = num_rounds
        return super().fit(num_rounds, timeout)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """
        Run one round of federated learning fit.

        Resets the client manager sampling at the beginning. Collects the results of fit for each
        client to be used later for generalization adjustment.

        Args:
            server_round: (int) the current server round it is in.
            timeout: (Optional[float]) the timeout to wait for clients to respond.

        Returns: (Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]) the results
            of fit_round, which consists of a tuple containing:
                1. The updated and aggregated parameters
                2. The aggregated metrics
                3. An instance of FitResultsAndFailures with the individual client's results and metrics.
        """

        # Resetting sampling and metrics
        self.client_manager().reset_sample()
        self.clients_metrics = []

        res_fit = super().fit_round(server_round, timeout)

        clients_proxies = self.client_manager().current_sample
        assert clients_proxies is not None

        # Collecting train metrics
        # should contain evaluation of the client's local model against its validation set
        if res_fit:
            # TODO what to do in case of failure?
            _, _, (results, failures) = res_fit
            assert len(results) == len(clients_proxies)

            self.results_and_failures = (results, failures)
            for i in range(len(results)):
                self.clients_metrics.append(
                    ClientMetrics(
                        cid=clients_proxies[i].cid,
                        train_metrics=results[i][1].metrics,
                    )
                )

        return res_fit

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """
        Run one round of federated learning evaluation.

        Collects the result of evaluate and perform the Domain Generalization algorithm by:
            1. Calculating the per-client Generalization Adaptation weights based on
                the evaluate and fit results
            2. Applying the clients' weights to each one of their results
            3. Running aggregation again with the weighted results
            4. Updating the parameters with the new aggregated parameters to be used
                in the next fit round

        Args:
            server_round: (int) the current server round it is in.
            timeout: (Optional[float]) the timeout to wait for clients to respond.

        Returns: (Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]) the results
            of evaluate_round, which consists of a tuple containing:
                1. The aggregated loss
                2. The aggregated metrics
                3. An instance of FitResultsAndFailures with the individual client's results and metrics.
        """
        res_eval = super().evaluate_round(server_round, timeout)

        clients_proxies = self.client_manager().current_sample
        assert clients_proxies is not None

        # Collecting evaluation metrics
        # which consists of evaluation of the global model against the client's validation set
        if res_eval:
            # TODO what to do in case of failure?
            loss_fed, _, (results, _) = res_eval
            for i in range(len(results)):
                assert clients_proxies[i].cid == self.clients_metrics[i].cid
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
        """
        Update the self.adjustment_weights dictionary by calculating the new weights
        based on the current server round, fit and evaluation metrics.

        Args:
            server_round: (int) the current server round.
        """
        clients_proxies = self.client_manager().current_sample
        assert clients_proxies is not None

        value_list = []
        # calculating local vs global metric difference
        for i in range(len(self.clients_metrics)):
            cid = clients_proxies[i].cid
            client_metrics = self.clients_metrics[i]

            assert client_metrics.cid == cid
            assert client_metrics.evaluation_metrics is not None and client_metrics.train_metrics is not None

            global_model_metric_value = client_metrics.evaluation_metrics[self.fairness_metric.metric_name]
            local_model_metric_value = client_metrics.train_metrics[self.fairness_metric.metric_name]
            assert isinstance(global_model_metric_value, float) and isinstance(local_model_metric_value, float)

            value_list.append(global_model_metric_value - local_model_metric_value)

            # initializing weight for this client
            if cid not in self.adjustment_weights:
                self.adjustment_weights[cid] = 1.0 / 3.0

        # calculating norm gap
        value_list_ndarray = np.array(value_list)
        norm_gap_list = value_list_ndarray / np.max(np.abs(value_list_ndarray))
        step_size = 1.0 / 3.0 * self.get_current_weight_step_size(server_round)

        # updating weights
        new_total_weight = 0.0
        for i in range(len(clients_proxies)):
            cid = clients_proxies[i].cid
            self.adjustment_weights[cid] += self.fairness_metric.signal * norm_gap_list[i] * step_size

            # weight clip
            clipped_weight = np.clip(self.adjustment_weights[cid], 0.0, 1.0)
            self.adjustment_weights[cid] = clipped_weight
            # TODO Question: should we sum all the weights or just the current clients' weights?
            new_total_weight += clipped_weight

        for cid in self.adjustment_weights:
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

    def apply_weights_to_results(self) -> None:
        """Apply self.adjustment_weights to self.results_and_failures."""
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
