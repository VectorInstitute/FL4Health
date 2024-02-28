from typing import Dict, Optional, Tuple

from flwr.common import Parameters
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.server import FitResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServer


class FedDGGAServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
    ) -> None:
        # TODO docstrings
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer, metrics_reporter)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        res_fit = super().fit_round(server_round, timeout)
        if res_fit:
            # import ipdb;ipdb.set_trace()

            # TODO what to do in case of failure?

            parameters_prime, _, (results, _) = res_fit

        return res_fit


class DeterministicFitAndEvaluateClientManager(SimpleClientManager):
    # TODO override sample
    pass
