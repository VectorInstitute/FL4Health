import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl4health.model_bases.fenda_base import (
    FendaGlobalModule,
    FendaHeadModule,
    FendaJoinMode,
    FendaLocalModule,
    FendaModel,
)
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger


class FendaTestClassifier(FendaHeadModule):
    def __init__(self, input_size: int, join_mode: FendaJoinMode) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(input_size, 2)

    def local_global_concat(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.fc1(input_tensor)
        return x


class LocalFendaTest(FendaLocalModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x


class GlobalFendaTest(FendaGlobalModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2 * 4 * 4)
        x = F.relu(self.fc1(x))
        return x


def test_fenda_join_and_layer_exchange() -> None:
    model = FendaModel(LocalFendaTest(), GlobalFendaTest(), FendaTestClassifier(6, FendaJoinMode.CONCATENATE))
    fenda_layers_to_exchange = sorted(model.layers_to_exchange())
    assert fenda_layers_to_exchange == [
        "global_module.conv1.bias",
        "global_module.conv1.weight",
        "global_module.fc1.bias",
        "global_module.fc1.weight",
    ]
    parameter_exchanger = FixedLayerExchanger(fenda_layers_to_exchange)
    parameters_to_exchange = parameter_exchanger.push_parameters(model)
    # 4 layers are expected, as weight and bias are separate for conv1 and fc1
    assert len(parameters_to_exchange) == 4
    for index, global_params in enumerate(model.global_module.state_dict().values()):
        assert np.array_equal(parameters_to_exchange[index], global_params.cpu().numpy())
    input = torch.ones((3, 1, 10, 10))
    # Test that concatenation produces the right output dimension
    output_shape = model(input).shape
    # Batch size
    assert output_shape[0] == 3
    # Output size
    assert output_shape[1] == 2
    # Test that summing produces the right output dimension
    model = FendaModel(LocalFendaTest(), GlobalFendaTest(), FendaTestClassifier(3, FendaJoinMode.SUM))
    output_shape = model(input).shape
    # Batch size
    assert output_shape[0] == 3
    # Output size
    assert output_shape[1] == 2
