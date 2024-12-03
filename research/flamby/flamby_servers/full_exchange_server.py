from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.servers.base_server import FlServer


class FullExchangeServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: Strategy | None = None,
        checkpoint_and_state_module: BaseServerCheckpointAndStateModule | None = None,
    ) -> None:
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            checkpoint_and_state_module=checkpoint_and_state_module,
        )
        # If parameter exchanger has been defined, it needs to be of type FullParameterExchanger
        if self.checkpoint_and_state_module.parameter_exchanger is not None:
            assert isinstance(self.checkpoint_and_state_module.parameter_exchanger, FullParameterExchanger)
