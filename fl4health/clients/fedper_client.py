from flwr.common.typing import Config

from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger


class FedPerClient(BasicClient):
    """
    Client to implement the FedPer method (https://arxiv.org/abs/1912.00818).

    Trains a global feature extractor shared by all clients through FedAvg and a private classifier that is unique to
    each client. The training is nearly identical to the ``BasicClient`` with the exception that our parameter
    exchanger needs to be a fixed layer exchanger that only exchanges the feature extraction base, which relies on the
    model being of type ``SequentiallySplitExchangeBaseModel``.
    """

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert isinstance(self.model, SequentiallySplitExchangeBaseModel), (
            "Models for FedPer must be of type SequentiallySplitExchangeBaseModel to facilitate partial weight "
            f"exchange. The current model is of type {type(self.model)}."
        )
        return FixedLayerExchanger(self.model.layers_to_exchange())
