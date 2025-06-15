from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from torch import nn

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServerWithCheckpointing


class FullExchangeServer(FlServerWithCheckpointing):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        strategy: Strategy | None = None,
        checkpointer: TorchModuleCheckpointer | None = None,
    ) -> None:
        # To help with model rehydration
        parameter_exchanger = FullParameterExchanger()
        super().__init__(
            client_manager=client_manager,
            parameter_exchanger=parameter_exchanger,
            model=model,
            strategy=strategy,
            checkpointer=checkpointer,
        )
