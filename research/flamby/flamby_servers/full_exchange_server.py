from typing import Optional

import torch.nn as nn
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer


class FullExchangeServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        model: Optional[nn.Module] = None,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[TorchModuleCheckpointer] = None,
    ) -> None:
        # To help with model rehydration
        parameter_exchanger = FullParameterExchanger()
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
        )
