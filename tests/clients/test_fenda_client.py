import copy

import pytest
import torch
from flwr.common import Config

from fl4health.clients.fenda_client import FendaClient
from fl4health.model_bases.fenda_base import FendaModel
from tests.clients.fixtures import get_fenda_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_getting_parameters(get_fenda_client: FendaClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_client = get_fenda_client
    config: Config = {}

    assert isinstance(fenda_client.model, FendaModel)
    params_local = [
        copy.deepcopy(val.cpu().numpy()) for _, val in fenda_client.model.local_module.state_dict().items()
    ]
    params_global = [
        copy.deepcopy(val.cpu().numpy()) for _, val in fenda_client.model.global_module.state_dict().items()
    ]
    fenda_client.get_parameters(config)

    assert isinstance(fenda_client.old_local_module, torch.nn.Module)
    for param in fenda_client.old_local_module.parameters():
        assert param.requires_grad is False

    old_local_module_params = [val.cpu().numpy() for _, val in fenda_client.old_local_module.state_dict().items()]
    for i in range(len(params_local)):
        assert (params_local[i] == old_local_module_params[i]).all()

    assert isinstance(fenda_client.old_global_module, torch.nn.Module)
    for param in fenda_client.old_global_module.parameters():
        assert param.requires_grad is False

    old_global_module_params = [val.cpu().numpy() for _, val in fenda_client.old_global_module.state_dict().items()]
    for i in range(len(params_global)):
        assert (params_global[i] == old_global_module_params[i]).all()


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_computting_contrastive_loss(get_fenda_client: FendaClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_client = get_fenda_client
    fenda_client.temperature = 0.5

    features = torch.tensor([[1, 1, 1]]).float()
    positive_pairs = torch.tensor([[1, 1, 1]]).float()
    negative_pairs = torch.tensor([[0, 0, 0]]).float()
    contrastive_loss = fenda_client.compute_contrastive_loss(features, positive_pairs, negative_pairs)

    assert contrastive_loss == pytest.approx(0.1269, rel=0.01)

    features = torch.tensor([[0, 0, 0]]).float()
    positive_pairs = torch.tensor([[1, 1, 1]]).float()
    negative_pairs = torch.tensor([[0, 0, 0]]).float()
    contrastive_loss = fenda_client.compute_contrastive_loss(features, positive_pairs, negative_pairs)

    assert contrastive_loss == pytest.approx(0.6931, rel=0.01)
