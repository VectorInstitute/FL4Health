from typing import Dict, Optional, Tuple

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.server import EvaluateResultsAndFailures
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.server.server import FlServer


class ScaffoldServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[BestMetricTorchCheckpointer] = None,
    ) -> None:
        self.client_model = client_model
        # To help with model rehydration
        model_size = len(self.client_model.state_dict())
        self.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        super().__init__(client_manager, strategy, checkpointer=checkpointer)

    def _hydrate_model_for_checkpointing(self) -> None:
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the control variates for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.client_model)

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
