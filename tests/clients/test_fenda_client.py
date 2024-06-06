import copy

import numpy as np
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
    # Setting the current server to 2, so that we don't set all of the weights.
    config: Config = {"current_server_round": 2}

    assert isinstance(fenda_client.model, FendaModel)
    # Save the original local module parameters before exchange
    old_params_local = [
        copy.deepcopy(val.cpu().numpy()) for _, val in fenda_client.model.first_feature_extractor.state_dict().items()
    ]
    # Get parameters that we send to the server for aggregation. Should be the params_global values.
    global_params_for_server = fenda_client.get_parameters(config)
    # Perturb the parameters to mimic server-side aggregation.
    global_params_from_server = [nd_array + 1.0 for nd_array in global_params_for_server]
    # Set the parameters on the client side to mimic communication.
    fenda_client.set_parameters(global_params_from_server, config, True)

    # Now check that the local parameters were not modified and the global parameters were.
    new_params_local = [
        val.cpu().numpy() for _, val in fenda_client.model.first_feature_extractor.state_dict().items()
    ]
    assert len(old_params_local) > 0
    assert len(new_params_local) > 0
    for old_layer_global, new_layer_local in zip(old_params_local, new_params_local):
        np.testing.assert_allclose(old_layer_global, new_layer_local, rtol=0, atol=1e-5)

    new_params_global = [
        val.cpu().numpy() for _, val in fenda_client.model.second_feature_extractor.state_dict().items()
    ]
    assert len(global_params_from_server) > 0
    assert len(new_params_global) > 0
    for layer_from_server, new_layer_global in zip(global_params_from_server, new_params_global):
        np.testing.assert_allclose(layer_from_server, new_layer_global, rtol=0, atol=1e-5)

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_computing_loss(get_fenda_client: FendaClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_client = get_fenda_client
    fenda_client.criterion = torch.nn.CrossEntropyLoss()

    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    training_loss = fenda_client.compute_training_loss(preds=preds, target=target, features={})
    evaluation_loss = fenda_client.compute_evaluation_loss(preds=preds, target=target, features={})

    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    # The evaluation and training losses should just be the vanilla cross-entropy losses
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(0.8132616, abs=0.0001) == training_loss.backward["backward"].item()
    # No additional losses should be computed for the FENDA client.
    assert len(training_loss.additional_losses) == 0
    assert len(evaluation_loss.additional_losses) == 0

    torch.seed()  # resetting the seed at the end, just to be safe
