import pytest
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.optimizers.optimizer import DPOptimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.scaffold_client import InstanceLevelPrivacyClient
from tests.clients.fixtures import get_client  # noqa


@pytest.mark.parametrize("type,model", [(InstanceLevelPrivacyClient, Net())])
def test_seed_setting(get_client: InstanceLevelPrivacyClient) -> None:  # noqa
    client = get_client
    assert client.seed == 2023


@pytest.mark.parametrize("type,model", [(InstanceLevelPrivacyClient, Net())])
def test_instance_level_client(get_client: InstanceLevelPrivacyClient) -> None:  # noqa
    client = get_client
    client.setup_opacus_objects()

    assert isinstance(client.model, GradSampleModule)
    assert isinstance(client.optimizer, DPOptimizer)
    assert isinstance(client.train_loader, DataLoader)
