from typing import Optional

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from research.flamby.flamby_servers.flamby_server import FlambyServer


class FedproxServer(FlambyServer):
    def __init__(
        self,
        client_manager: ClientManager,
        client_model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[BestMetricTorchCheckpointer] = None,
    ) -> None:
        super().__init__(client_manager, client_model, strategy, checkpointer=checkpointer)
        # To help with model rehydration
        self.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerFedProx())

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the extra fedprox variable for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.client_model)
        return self.client_model
