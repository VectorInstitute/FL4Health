from typing import Optional

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.server.base_server import FlServerWithCheckpointing


class ScaffoldServer(FlServerWithCheckpointing[FullParameterExchangerWithPacking]):
    def __init__(
        self,
        client_manager: ClientManager,
        model: Optional[nn.Module] = None,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        assert model is not None
        # To help with model rehydration
        model_size = len(model.state_dict())
        parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        super().__init__(
            client_manager=client_manager,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
        )

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        assert (
            self.server_model is not None
        ), "Model hydration has been called but no server_model is defined to hydrate"
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the control variates for checkpointing.
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model
