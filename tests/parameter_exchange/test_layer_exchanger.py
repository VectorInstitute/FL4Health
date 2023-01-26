import numpy as np
import torch
import torch.nn as nn

from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.independent_layer = nn.Linear(4, 4, bias=False)
        self.shared_layer = nn.Linear(3, 3, bias=False)


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
