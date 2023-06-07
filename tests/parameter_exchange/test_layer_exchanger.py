import numpy as np
import torch
import torch.nn as nn

from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger, NormDriftLayerExchanger


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.independent_layer = nn.Linear(4, 4, bias=False)
        self.shared_layer = nn.Linear(3, 3, bias=False)


class ToyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 4 * 4, 120, bias=False)


def test_layer_parameter_exchange() -> None:
    model = LinearModel()
    # fill model weights with different constants
    nn.init.constant_(model.independent_layer.weight, 1.0)
    nn.init.constant_(model.shared_layer.weight, 2.0)
    exchanger = FixedLayerExchanger(["shared_layer.weight"])
    shared_layer_list = exchanger.push_parameters(model)
    assert len(shared_layer_list) == 1
    shared_layer_exchanged = shared_layer_list[0]
    assert np.array_equal(shared_layer_exchanged, 2.0 * np.ones((3, 3)))
    exchanger.pull_parameters([2.0 * p for p in shared_layer_list], model)
    assert torch.all(torch.eq(model.shared_layer.weight, 4.0 * torch.ones((3, 3))))
    assert torch.all(torch.eq(model.independent_layer.weight, torch.ones((4, 4))))


def test_norm_drift_layer_exchange() -> None:
    initial_model = ToyConvNet()
    model_to_exchange = ToyConvNet()

    nn.init.constant_(initial_model.conv1.weight, 0)
    nn.init.constant_(initial_model.conv2.weight, 0)
    nn.init.constant_(initial_model.fc1.weight, 0)

    nn.init.constant_(model_to_exchange.conv1.weight, 1 / 20000)
    nn.init.constant_(model_to_exchange.conv2.weight, 1 / 20000)
    nn.init.constant_(model_to_exchange.fc1.weight, 1)

    threshold = 2

    exchanger = NormDriftLayerExchanger(initial_model, threshold)

    layers_with_names_to_exchange = exchanger.push_parameters(model_to_exchange)
    layers_to_exchange, layer_names = exchanger.unpack_parameters(layers_with_names_to_exchange)
    assert len(layers_to_exchange) == 1
    assert len(layer_names) == 1
