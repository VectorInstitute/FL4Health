from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.clients.scaffold_client import ScaffoldClient
from fl4health.parameter_exchange.packing_exchanger import (
    ParameterExchangerWithClippingBit,
    ParameterExchangerWithControlVariates,
)
from fl4health.utils.metrics import Accuracy


@pytest.fixture
def get_client(type: type, model: nn.Module) -> NumpyFlClient:
    client: NumpyFlClient
    if type == ScaffoldClient:
        client = ScaffoldClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.learning_rate_local = 0.01
        client.parameter_exchanger = ParameterExchangerWithControlVariates()
    elif type == FedProxClient:
        client = FedProxClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.parameter_exchanger = ParameterExchangerWithClippingBit()
    else:
        raise ValueError(f"{str(type)} is not a valid client type")

    client.model = model
    return client
