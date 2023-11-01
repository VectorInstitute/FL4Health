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
    new_prams = [layer_weights + 0.1 for layer_weights in params]
    moon_client.set_parameters(new_prams, config)

    old_model_params = [val.cpu().numpy() for _, val in moon_client.old_models_list[-1].state_dict().items()]
    global_model_params = [val.cpu().numpy() for _, val in moon_client.global_model.state_dict().items()]

    for i in range(len(params)):
        assert (params[i] == old_model_params[i]).all()
        assert (new_prams[i] == global_model_params[i]).all()
