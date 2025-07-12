from flwr.common.typing import Config

from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.layer_exchanger import LayerExchangerWithExclusions


class FedBnClient(BasicClient):
    """
    This class serves as a sparse interface for clients aiming to leverage the FedBN method
    (https://arxiv.org/abs/2102.07623) or any other approach that excludes specific types of model layers during
    parameter exchange. This class simply ensures that the user has overridden the ``get_parameter_exchanger``
    properly.

    For example, in FedBN, batch normalization layers are excluded from exchange with the server
    but all other layers flow through and are aggregated via whatever strategy the server is implementing. An example
    of this where one wants to exclude 2D batch normalization layers during exchange is
    ``LayerExchangerWithExclusions(self.model, {nn.BatchNorm2d})``, where the model is provided so that the exchanger
    can identify the appropriate layers to leave out.
    """

    def setup_client(self, config: Config) -> None:
        super().setup_client(config=config)
        assert isinstance(self.parameter_exchanger, LayerExchangerWithExclusions), (
            "For FedBnClients the parameter exchanger must be of type LayerExchangerWithExclusions "
            f"but got {type(self.parameter_exchanger)}. If you haven't already, override the get_parameter_exchanger "
            "function in your class."
        )
        return super().setup_client(config)
