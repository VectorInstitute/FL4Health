from collections import OrderedDict

import pytest
import torch
from flwr.common import Config

from fl4health.clients.ditto_client import DittoClient
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import SmallCnn


@pytest.mark.parametrize("type,model", [(DittoClient, SmallCnn())])
def test_setting_initial_weights(get_client: DittoClient) -> None:  # noqa
    torch.manual_seed(42)
    ditto_client = get_client
    ditto_client.global_model = SmallCnn()
    ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {"current_server_round": 1}

    drift_penalty_weight = 10.0
    params = [val.cpu().numpy() + 1.0 for _, val in ditto_client.model.state_dict().items()]
    packed_params = ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    ditto_client.set_parameters(packed_params, config, fitting_round=True)
    ditto_client.update_before_train(1)

    # First fitting round we should set both the global and local models to params and store the global model values
    assert ditto_client.drift_penalty_tensors is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases, fc1 weights, biases (so 6 total)
    assert len(ditto_client.drift_penalty_tensors) == 6
    # Make sure that we saved the right parameters
    for layer_init_global_tensor, layer_params in zip(ditto_client.drift_penalty_tensors, params):
        assert pytest.approx(torch.sum(layer_init_global_tensor - layer_params), abs=0.0001) == 0.0
    # Make sure the global model was set correctly
    for global_model_layer_params, layer_params in zip(ditto_client.global_model.parameters(), params):
        assert pytest.approx(torch.sum(global_model_layer_params.detach() - layer_params), abs=0.0001) == 0.0
    # Make sure the local model was set correctly
    for local_model_layer_params, layer_params in zip(ditto_client.model.parameters(), params):
        assert pytest.approx(torch.sum(local_model_layer_params.detach() - layer_params), abs=0.0001) == 0.0

    assert ditto_client.drift_penalty_weight == 10.0

    # Now we update in a later round
    config = {"current_server_round": 2}
    params = [param + 1.0 for param in params]
    packed_params = ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight + 1.0)
    ditto_client.set_parameters(packed_params, config, fitting_round=True)
    ditto_client.update_before_train(2)
    # Make sure that we saved the right parameters
    for layer_init_global_tensor, layer_params in zip(ditto_client.drift_penalty_tensors, params):
        assert pytest.approx(torch.sum(layer_init_global_tensor - layer_params), abs=0.0001) == 0.0
    # Make sure the global model was set correctly
    for global_model_layer_params, layer_params in zip(ditto_client.global_model.parameters(), params):
        assert pytest.approx(torch.sum(global_model_layer_params.detach() - layer_params), abs=0.0001) == 0.0
    # Make sure the local model WAS NOT UPDATED
    for local_model_layer_params, layer_params in zip(ditto_client.model.parameters(), params):
        assert pytest.approx(torch.sum(local_model_layer_params.detach() - layer_params), abs=0.0001) != 0.0

    assert ditto_client.drift_penalty_weight == 11.0


@pytest.mark.parametrize("type,model", [(DittoClient, SmallCnn())])
def test_forming_ditto_loss(get_client: DittoClient) -> None:  # noqa
    torch.manual_seed(42)
    ditto_client = get_client
    ditto_client.global_model = SmallCnn()
    ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    drift_penalty_weight = 1.0
    config: Config = {"current_server_round": 2}

    params = [val.cpu().numpy() + 0.1 for _, val in ditto_client.model.state_dict().items()]
    packed_params = ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    ditto_client.set_parameters(packed_params, config, fitting_round=True)
    ditto_client.update_before_train(4)

    ditto_loss = ditto_client.penalty_loss_function(
        ditto_client.model, ditto_client.drift_penalty_tensors, ditto_client.drift_penalty_weight
    )

    assert ditto_client.drift_penalty_weight == 1.0
    assert pytest.approx(ditto_loss.detach().item(), abs=0.02) == (ditto_client.drift_penalty_weight / 2.0) * (
        1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32
    )


@pytest.mark.parametrize("type,model", [(DittoClient, SmallCnn())])
def test_compute_loss(get_client: DittoClient) -> None:  # noqa
    torch.manual_seed(42)
    ditto_client = get_client
    ditto_client.global_model = SmallCnn()
    ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {"current_server_round": 2}
    ditto_client.criterion = torch.nn.CrossEntropyLoss()
    drift_penalty_weight = 1.0

    params = [val.cpu().numpy() for _, val in ditto_client.model.state_dict().items()]
    packed_params = ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    ditto_client.set_parameters(packed_params, config, fitting_round=True)
    ditto_client.update_before_train(4)

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]

    params_dict = zip(ditto_client.model.state_dict().keys(), perturbed_params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    ditto_client.model.load_state_dict(state_dict, strict=True)

    preds = {"global": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "local": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    training_loss = ditto_client.compute_training_loss(preds, {}, target)
    ditto_client.global_model.eval()
    ditto_client.model.eval()
    evaluation_loss = ditto_client.compute_evaluation_loss(preds, {}, target)
    assert isinstance(training_loss.backward, dict)
    assert pytest.approx(54.7938, abs=0.01) == training_loss.backward["backward"].item()
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()
