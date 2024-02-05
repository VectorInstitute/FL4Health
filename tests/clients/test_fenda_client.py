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
def test_computing_contrastive_loss(get_fenda_client: FendaClient) -> None:  # noqa
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


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_computing_loss(get_fenda_client: FendaClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_client = get_fenda_client
    fenda_client.temperature = 0.5
    fenda_client.perfcl_loss_weights = (1.0, 1.0)
    fenda_client.criterion = torch.nn.CrossEntropyLoss()

    local_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    old_local_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    old_global_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    aggregated_global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    features = {
        "local_features": local_features,
        "old_local_features": old_local_features,
        "global_features": global_features,
        "old_global_features": old_global_features,
        "aggregated_global_features": aggregated_global_features,
    }

    training_loss = fenda_client.compute_training_loss(preds=preds, target=target, features=features)
    evaluation_loss = fenda_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(3.0671176, abs=0.0001) == training_loss.backward["backward"].item()
    assert training_loss.additional_losses == evaluation_loss.additional_losses
    assert training_loss.additional_losses["loss"] == evaluation_loss.checkpoint.item()
    assert training_loss.additional_losses["total_loss"] == training_loss.backward["backward"].item()

    auxiliary_loss_total = (training_loss.backward["backward"] - evaluation_loss.checkpoint).item()
    contrastive_minimize = training_loss.additional_losses["contrastive_loss_minimize"].item()
    contrastive_maximize = training_loss.additional_losses["contrastive_loss_maximize"].item()
    assert pytest.approx(auxiliary_loss_total, abs=0.001) == (contrastive_minimize + contrastive_maximize)
