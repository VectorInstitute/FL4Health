import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.parallel_split_models import ParallelFeatureJoinMode, ParallelSplitHeadModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger


class FendaTestClassifier(ParallelSplitHeadModule):
    def __init__(self, input_size: int, join_mode: ParallelFeatureJoinMode) -> None:
        super().__init__(join_mode)
        self.fc1 = nn.Linear(input_size, 2)

    def parallel_output_join(self, local_tensor: torch.Tensor, global_tensor: torch.Tensor) -> torch.Tensor:
        # Assuming tensors are "batch first" so join column-wise
        return torch.concat([local_tensor, global_tensor], dim=1)

    def head_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.fc1(input_tensor)


class LocalFendaTest(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2 * 4 * 4)
        return F.relu(self.fc1(x))


class GlobalFendaTest(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2 * 4 * 4)
        return F.relu(self.fc1(x))


def test_fenda_join_and_layer_exchange() -> None:
    model = FendaModel(
        LocalFendaTest(), GlobalFendaTest(), FendaTestClassifier(6, ParallelFeatureJoinMode.CONCATENATE)
    )
    fenda_layers_to_exchange = sorted(model.layers_to_exchange())
    assert fenda_layers_to_exchange == [
        "second_feature_extractor.conv1.bias",
        "second_feature_extractor.conv1.weight",
        "second_feature_extractor.fc1.bias",
        "second_feature_extractor.fc1.weight",
    ]
    parameter_exchanger = FixedLayerExchanger(fenda_layers_to_exchange)
    parameters_to_exchange = parameter_exchanger.push_parameters(model)
    # 4 layers are expected, as weight and bias are separate for conv1 and fc1
    assert len(parameters_to_exchange) == 4
    model_state_dict = model.state_dict()
    for layer_name, layer_parameters in zip(fenda_layers_to_exchange, parameters_to_exchange):
        assert np.array_equal(layer_parameters, model_state_dict[layer_name])

    # Insert the weights back into another model
    model_2 = FendaModel(
        LocalFendaTest(), GlobalFendaTest(), FendaTestClassifier(6, ParallelFeatureJoinMode.CONCATENATE)
    )
    parameter_exchanger.pull_parameters(parameters_to_exchange, model_2)
    for layer_name, layer_parameters in model_2.state_dict().items():
        if layer_name in fenda_layers_to_exchange:
            assert np.array_equal(layer_parameters, model_state_dict[layer_name])

    input = torch.ones((3, 1, 10, 10))
    # Test that concatenation produces the right output dimension
    preds, _ = model(input)
    output_shape = preds["prediction"].shape
    # Batch size
    assert output_shape[0] == 3
    # Output size
    assert output_shape[1] == 2
    # Test that summing produces the right output dimension
    model = FendaModel(LocalFendaTest(), GlobalFendaTest(), FendaTestClassifier(3, ParallelFeatureJoinMode.SUM))
    preds, _ = model(input)
    output_shape = preds["prediction"].shape
    # Batch size
    assert output_shape[0] == 3
    # Output size
    assert output_shape[1] == 2
