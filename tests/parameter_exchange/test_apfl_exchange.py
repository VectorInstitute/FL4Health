import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.apfl_base import ApflModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2 * 4 * 4)
        return F.relu(self.fc1(x))


def test_apfl_layer_exchange() -> None:
    model = ApflModule(ToyModel())
    apfl_layers_to_exchange = sorted(model.layers_to_exchange())
    assert apfl_layers_to_exchange == [
        "global_model.conv1.bias",
        "global_model.conv1.weight",
        "global_model.fc1.bias",
        "global_model.fc1.weight",
    ]
    parameter_exchanger = FixedLayerExchanger(apfl_layers_to_exchange)
    parameters_to_exchange = parameter_exchanger.push_parameters(model)
    # 4 layers are expected, as weight and bias are separate for conv1 and fc1
    assert len(parameters_to_exchange) == 4
    model_state_dict = model.state_dict()
    for layer_name, layer_parameters in zip(apfl_layers_to_exchange, parameters_to_exchange):
        assert np.array_equal(layer_parameters, model_state_dict[layer_name])

    # Insert the weights back into another model
    model_2 = ApflModule(ToyModel())
    parameter_exchanger.pull_parameters(parameters_to_exchange, model_2)
    for layer_name, layer_parameters in model_2.state_dict().items():
        if layer_name in apfl_layers_to_exchange:
            assert np.array_equal(layer_parameters, model_state_dict[layer_name])

    input = torch.ones((3, 1, 10, 10))
    # APFL returns a dict with personal, global and local predictions
    # Assert return values of each prediction type is correct dim
    preds = model(input)
    personal_shape = preds["personal"].shape
    # Batch size
    assert personal_shape[0] == 3
    # Output size
    assert personal_shape[1] == 3

    global_shape = preds["global"].shape
    # Batch size
    assert global_shape[0] == 3
    # Output size
    assert global_shape[1] == 3

    local_shape = preds["local"].shape
    # Batch size
    assert local_shape[0] == 3
    # Output size
    assert local_shape[1] == 3
