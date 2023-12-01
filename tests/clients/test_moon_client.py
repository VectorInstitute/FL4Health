import copy

import pytest
import torch
from flwr.common import Config

from fl4health.clients.moon_client import MoonClient
from fl4health.model_bases.moon_base import MoonModel
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_setting_parameters(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    config: Config = {}

    params = [copy.deepcopy(val.cpu().numpy()) for _, val in moon_client.model.state_dict().items()]
    new_params = [layer_weights + 0.1 for layer_weights in params]
    moon_client.set_parameters(new_params, config)

    old_model_params = [val.cpu().numpy() for _, val in moon_client.old_models_list[-1].state_dict().items()]
    global_model_params = [val.cpu().numpy() for _, val in moon_client.global_model.state_dict().items()]

    for i in range(len(params)):
        assert (params[i] == old_model_params[i]).all()
        assert (new_params[i] == global_model_params[i]).all()


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_contrastive_loss(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client

    global_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, -1.0]]])
    previous_local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    # Default temperature is 0.5
    contrastive_loss = moon_client.get_contrastive_loss(
        features=local_features.reshape(len(local_features), -1),
        global_features=global_features.reshape(len(global_features), -1),
        old_features=previous_local_features.reshape(1, len(previous_local_features), -1),
    )

    assert pytest.approx(0.837868, abs=0.0001) == contrastive_loss


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_compute_loss(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    # Dummy to ensure the compute loss function in the moon client is executed
    moon_client.old_models_list = [moon_client.model]
    moon_client.contrastive_weight = 1.0
    moon_client.criterion = torch.nn.CrossEntropyLoss()

    global_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[2.0, 1.0], [1.0, -1.0]]])
    previous_local_features = torch.tensor([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    features = {
        "features": local_features.reshape(len(local_features), -1),
        "global_features": global_features.reshape(len(global_features), -1),
        "old_features": previous_local_features.reshape(1, len(previous_local_features), -1),
    }

    loss = moon_client.compute_loss(preds=preds, features=features, target=target)

    assert pytest.approx(0.8132616, abs=0.0001) == loss.checkpoint.item()
    assert pytest.approx(0.837868, abs=0.0001) == loss.additional_losses["contrastive_loss"]
    assert pytest.approx(0.837868 + 0.8132616, abs=0.0001) == loss.backward.item()
