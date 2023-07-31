from typing import Optional

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from research.flamby.flamby_servers.flamby_server import FlambyServer


class FullExchangeServer(FlambyServer):
    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[BestMetricTorchCheckpointer] = None,
    ) -> None:
        self.client_model = client_model
        # To help with model rehydration
        self.parameter_exchanger = FullParameterExchanger()
        super().__init__(client_manager, client_model, strategy, checkpointer=checkpointer)

    def _hydrate_model_for_checkpointing(self) -> None:
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.client_model)
