from typing import Optional

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from fl4health.server.base_server import FlServerWithCheckpointing


class FedProxServer(FlServerWithCheckpointing[ParameterExchangerWithPacking]):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        # To help with model rehydration
        parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerFedProx())
        super().__init__(
            client_manager=client_manager,
            model=model,
            parameter_exchanger=parameter_exchanger,
            strategy=strategy,
            checkpointer=checkpointer,
        )

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        # Overriding the standard hydration method to account for the unpacking
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the extra fedprox variable for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model
