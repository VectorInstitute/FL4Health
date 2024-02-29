from typing import Dict, List, Optional, Tuple

from flwr.common import Parameters
from flwr.common.typing import Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServer


class FixedSamplingClientManager(SimpleClientManager):
    """Keeps sampling fixed until it's asked to reset"""

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
    def __init__(
        self,
        train_metrics: Optional[Dict[str, Scalar]] = None,
        evaluation_metrics: Optional[Dict[str, Scalar]] = None,
    ):
        self.train_metrics = train_metrics
        self.evaluation_metrics = evaluation_metrics

    def __repr__(self) -> str:
        return f"train_metrics: {self.train_metrics}, evaluation_metrics: {self.evaluation_metrics}"


class FedDGGAServer(FlServer):
    def __init__(
        self,
        client_manager: FixedSamplingClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        # TODO docstrings
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer, metrics_reporter)
        self.clients_metrics: List[ClientMetrics] = []

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        # TODO docstrings

        client_manager = self.client_manager()
        assert isinstance(client_manager, FixedSamplingClientManager), (
            "Client manager is not of type ClientManagerWithFixedSample" f"({type(client_manager)})"
        )
        client_manager.reset_sample()
        self.clients_metrics = []

        res_fit = super().fit_round(server_round, timeout)
        if res_fit:
            # TODO what to do in case of failure?
            _, _, (results, _) = res_fit
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
        if res_eval:
            # TODO what to do in case of failure?
            _, _, (results, _) = res_eval
            for i in range(len(results)):
                self.clients_metrics[i].evaluation_metrics = results[i][1].metrics

        return res_eval
