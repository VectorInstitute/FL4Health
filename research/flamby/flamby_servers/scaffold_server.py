from typing import Optional

import torch.nn as nn
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.servers.base_server import FlServer


class ScaffoldServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
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
            fl_config=fl_config,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
        )

    def _hydrate_model_for_checkpointing(self) -> None:
        assert self.server_model is not None, (
            "Model hydration has been called but no server_model is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a torch architecture is "
            "possible and desired"
        )
        assert self.parameter_exchanger is not None, (
            "Model hydration has been called but no parameter_exchanger is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a parameter exchanger is "
            "possible and desired"
        )
        packed_parameters = parameters_to_ndarrays(self.parameters)
        # Don't need the control variates for checkpointing.
        assert isinstance(self.parameter_exchanger, FullParameterExchangerWithPacking)
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
