import copy
from pathlib import Path

import pytest
import torch
from flwr.common.typing import Config
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.scaffold_client import InstanceLevelDpClient
from fl4health.metrics import Accuracy
from fl4health.utils.dataset import BaseDataset
from fl4health.utils.privacy_utilities import privacy_validate_and_fix_modules
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import MnistNetWithBnAndFrozen, Net


class DummyDataset(BaseDataset):
    def __init__(self, data_size: int = 100) -> None:
        # 100 is the number of samples
        self.data = torch.randn(100, data_size)
        self.targets = torch.randint(5, (data_size,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


class ClientForTest(InstanceLevelDpClient):
    def get_model(self, config: Config) -> nn.Module:
        model = MnistNetWithBnAndFrozen(freeze_cnn_layer=True).to(self.device)
        model.bn.weight = nn.Parameter(10 * torch.ones_like(model.bn.weight))
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


@pytest.mark.parametrize("type,model", [(InstanceLevelDpClient, Net())])
def test_instance_level_client(get_client: InstanceLevelDpClient) -> None:  # noqa
    client = get_client
    client.setup_opacus_objects({})

    assert isinstance(client.model, GradSampleModule)
    assert isinstance(client.optimizers["global"], DPOptimizer)
    assert isinstance(client.train_loader, DataLoader)


def test_privacy_validate_and_fix() -> None:
    # Get a network where opacus needs to replace the batch norms
    model: nn.Module = MnistNetWithBnAndFrozen(True)
    model, reinitialize_optimizer = privacy_validate_and_fix_modules(model)

    # We should need to reinitialize the optimizer parameters
    assert reinitialize_optimizer
    # The batch norm in the model should have been replaced with a GroupNorm
    assert isinstance(model.bn, torch.nn.modules.normalization.GroupNorm)


def test_instance_level_client_with_changes() -> None:  # noqa
    client = ClientForTest(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = client.get_model({})
    client.noise_multiplier = 1.0
    client.clipping_bound = 5.0
    client.set_optimizer({})
    original_optimizers = copy.deepcopy(client.optimizers)
    client.train_loader = DataLoader(DummyDataset(), batch_size=10)
    client.setup_opacus_objects({})

    # We should properly replace the model batch norm layer with GroupNorm
    assert isinstance(client.model.bn, torch.nn.modules.normalization.GroupNorm)
    original_param_objects = original_optimizers["global"].param_groups[0]["params"]
    new_param_objects = client.optimizers["global"].param_groups[0]["params"]
    # All layers should be identical except the new GroupNorm weight vs. the old BatchNorm weight, which we set to 10
    # The default value for GroupNorm is 1.0
    assert len(original_param_objects) == len(new_param_objects)
    for index, (original_params, new_params) in enumerate(zip(original_param_objects, new_param_objects), start=0):
        if index != 4:
            assert torch.all(original_params.data.eq(new_params.data))
        else:
            assert not torch.all(original_params.data.eq(new_params.data))
