from logging import INFO
from typing import Dict, Optional, Tuple

from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.server.server import FlServer


class PersonalServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        # Personal approaches don't train a "server" model. Rather, each client trains a client specific model with
        # some globally shared weights. So we don't checkpoint a global model
        super().__init__(client_manager, strategy, checkpointer=None)
        self.best_aggregated_loss: Optional[float] = None

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        # loss_aggregated is the aggregated validation per step loss
        # aggregated over each client (weighted by num examples)
        eval_round_results = super().evaluate_round(server_round, timeout)
        assert eval_round_results is not None
        loss_aggregated, metrics_aggregated, (results, failures) = eval_round_results
        assert loss_aggregated is not None

        if self.best_aggregated_loss:
            if self.best_aggregated_loss >= loss_aggregated:
                log(
                    INFO,
                    f"Best Aggregated Loss: {self.best_aggregated_loss} "
                    f"is larger than current aggregated loss: {loss_aggregated}",
                )
                self.best_aggregated_loss = loss_aggregated
            else:
                log(
                    INFO,
                    f"Best Aggregated Loss: {self.best_aggregated_loss} "
                    f"is smaller than current aggregated loss: {loss_aggregated}",
                )
        else:
            log(INFO, f"Saving Best Aggregated Loss: {loss_aggregated} as it is currently None")
            self.best_aggregated_loss = loss_aggregated

        return loss_aggregated, metrics_aggregated, (results, failures)
