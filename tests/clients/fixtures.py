from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.clients.scaffold_client import ScaffoldClient
from fl4health.parameter_exchange.packing_exchanger import (
    ParameterExchangerWithClippingBit,
    ParameterExchangerWithControlVariates,
)
from fl4health.utils.metrics import Accuracy


class TestCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x


class LinearTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


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
