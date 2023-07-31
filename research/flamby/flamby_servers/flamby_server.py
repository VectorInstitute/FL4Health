from typing import Dict, Optional, Tuple

import torch.nn as nn
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.server.server import FlServer


class FlambyServer(FlServer):
    """
    The FlambyServer is used for FL approaches that have a sense of a GLOBAL model that is checkpointed on the server
    side of the FL communcation framework. That is, a model that is to be shared among all clients. This is distinct
    from strictly PERSONAL model approaches like APFL or FENDA where each client will have its own model that is
    specific to its own training. Personal models may have shared components but the full model is specific to each
    client. These servers are implemented in the PersonalServer class.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[BestMetricTorchCheckpointer] = None,
    ) -> None:
        self.client_model = client_model
        super().__init__(client_manager, strategy, checkpointer=checkpointer)

    def _maybe_checkpoint(self, checkpoint_metric: float) -> None:
        if self.checkpointer:
            self._hydrate_model_for_checkpointing()
            self.checkpointer.maybe_checkpoint(self.client_model, checkpoint_metric)

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
        self._maybe_checkpoint(loss_aggregated)

        return loss_aggregated, metrics_aggregated, (results, failures)
