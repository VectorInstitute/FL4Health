import numpy as np
import pytest
import torch
from flwr.common.typing import Config

from fl4health.clients.apfl_client import ApflClient
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.apfl_base import ApflModule
from tests.clients.fixtures import get_apfl_client, get_basic_client  # noqa
from tests.test_utils.models_for_test import SingleLayerWithSeed, ToyConvNet


def to_numpy_clone(a: torch.Tensor) -> np.ndarray:
    return a.detach().cpu().clone().numpy()


@pytest.mark.parametrize("model", [ToyConvNet()])
def test_set_parameters_basic_client(get_basic_client: BasicClient) -> None:  # noqa
    client = get_basic_client
    client_state_dict = client.model.state_dict()
    old_model_state = [to_numpy_clone(state) for state in client_state_dict.values()]
    new_model_state_dict = ToyConvNet().state_dict()
    new_model_state = [to_numpy_clone(state) for state in new_model_state_dict.values()]
    new_model_cnn_state = [
        to_numpy_clone(new_model_state_dict["conv1.weight"]),
        to_numpy_clone(new_model_state_dict["conv2.weight"]),
    ]
    new_model_fc_state = [
        to_numpy_clone(new_model_state_dict["fc1.weight"]),
        to_numpy_clone(new_model_state_dict["fc2.weight"]),
    ]
    config: Config = {"current_server_round": 1}
    client.set_parameters(new_model_state, config, fitting_round=True)

    # Model state should match the new model state and be different from the old model state,
    # including the linear layers
    client_model_state = [to_numpy_clone(state) for state in client.model.state_dict().values()]
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(new_model_state, client_model_state))
    assert all(not np.allclose(a, b, atol=0.001) for a, b in zip(old_model_state, client_model_state))

    # Get the CNN parameters from the client, modify them by adding 1 to all weights, and place back in
    cnn_parameters_only = client.get_parameters({})
    # length of the parameters should be two (corresponding to the two cnn layers)
    assert len(cnn_parameters_only) == 2

    cnn_parameters_only = [weights + 1.0 for weights in cnn_parameters_only]
    config["current_server_round"] = 2
    client.set_parameters(cnn_parameters_only, config, fitting_round=True)
    client_state_dict = client.model.state_dict()
    client_model_cnn_state = [
        to_numpy_clone(client_state_dict["conv1.weight"]),
        to_numpy_clone(client_state_dict["conv2.weight"]),
    ]
    client_model_fc_state = [
        to_numpy_clone(client_state_dict["fc1.weight"]),
        to_numpy_clone(client_state_dict["fc2.weight"]),
    ]
    # CNN state should have been reset to the old model, but the FC layers should still be those of the "new" model
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(cnn_parameters_only, client_model_cnn_state))
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(new_model_fc_state, client_model_fc_state))
    # Alternatively the CNN states should no longer be the "new" model weights and the old model fc states should not
    # match
    assert all(not np.allclose(a, b, atol=0.001) for a, b in zip(new_model_cnn_state, client_model_cnn_state))


@pytest.mark.parametrize("type,model", [(ApflClient, SingleLayerWithSeed())])
def test_set_parameters_apfl_client(get_apfl_client: ApflClient) -> None:  # noqa
    client = get_apfl_client
    model_to_insert_state_dict = ApflModule(SingleLayerWithSeed(33)).state_dict()
    model_state_to_insert = [to_numpy_clone(state) for state in model_to_insert_state_dict.values()]
    model_to_insert_local_params = to_numpy_clone(model_to_insert_state_dict["local_model.linear.weight"])
    model_to_insert_global_params = to_numpy_clone(model_to_insert_state_dict["global_model.linear.weight"])

    config: Config = {"current_server_round": 1}

    # First time we're initializing the model, so all parameters should be initialized
    client.set_parameters(model_state_to_insert, config, fitting_round=True)
    current_model_state = [to_numpy_clone(state) for state in client.model.state_dict().values()]
    # The whole model should be initialized with model_state_to_insert
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(model_state_to_insert, current_model_state))

    # Only the global linear layer should be replaced
    # Get the CNN parameters from the client, modify them by adding 1 to all weights, and place back in
    global_layer_only = client.get_parameters({})
    # length of the parameters should be one (corresponding to the one global layer)
    assert len(global_layer_only) == 1

    global_layer_only = [weights + 1.0 for weights in global_layer_only]
    config["current_server_round"] = 2
    client.set_parameters(global_layer_only, config, fitting_round=True)
    client_state_dict = client.model.state_dict()
    client_model_local_state = to_numpy_clone(client_state_dict["local_model.linear.weight"])
    client_model_global_state = to_numpy_clone(client_state_dict["global_model.linear.weight"])
    # The local model should be unchanged
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(client_model_local_state, model_to_insert_local_params))
    # The global model should be updated, note global_layer_only is a list (so extract the only entry for comparison)
    assert all(np.allclose(a, b, atol=0.001) for a, b in zip(client_model_global_state, global_layer_only[0]))
    assert all(
        not np.allclose(a, b, atol=0.001) for a, b in zip(client_model_global_state, model_to_insert_global_params)
    )
