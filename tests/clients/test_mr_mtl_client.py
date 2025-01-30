from collections import OrderedDict

import pytest
import torch
from flwr.common import Config

from fl4health.clients.mr_mtl_client import MrMtlClient
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import SmallCnn


@pytest.mark.parametrize("type,model", [(MrMtlClient, SmallCnn())])
def test_setting_global_weights(get_client: MrMtlClient) -> None:  # noqa
    torch.manual_seed(42)
    mr_mtl_client = get_client
    mr_mtl_client.initial_global_model = SmallCnn()
    mr_mtl_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {}
    drift_penalty_weight = 10.0

    old_params = [val.cpu().numpy() for _, val in mr_mtl_client.model.state_dict().items()]
    params = [val.cpu().numpy() + 1.0 for _, val in mr_mtl_client.model.state_dict().items()]
    packed_params = mr_mtl_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    mr_mtl_client.set_parameters(packed_params, config, fitting_round=True)

    # We should set only init global model to params and store the global model values
    # Make sure that we saved the right parameters
    for layer_init_global_tensor, layer_params in zip(mr_mtl_client.initial_global_model.parameters(), params):
        assert pytest.approx(torch.sum(layer_init_global_tensor.detach() - layer_params), abs=0.0001) == 0.0

    # Make sure the local model was kept same
    for local_model_layer_params, layer_params in zip(mr_mtl_client.model.parameters(), old_params):
        assert pytest.approx(torch.sum(local_model_layer_params.detach() - layer_params), abs=0.0001) == 0.0


@pytest.mark.parametrize("type,model", [(MrMtlClient, SmallCnn())])
def test_forming_mr_loss(get_client: MrMtlClient) -> None:  # noqa
    torch.manual_seed(42)
    mr_mtl_client = get_client
    mr_mtl_client.initial_global_model = SmallCnn()
    mr_mtl_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {}
    drift_penalty_weight = 10.0

    params = [val.cpu().numpy() + 0.1 for _, val in mr_mtl_client.model.state_dict().items()]
    packed_params = mr_mtl_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    mr_mtl_client.set_parameters(packed_params, config, fitting_round=True)
    mr_mtl_client.update_before_train(4)

    mr_mtl_loss = mr_mtl_client.penalty_loss_function(
        mr_mtl_client.model, mr_mtl_client.drift_penalty_tensors, mr_mtl_client.drift_penalty_weight
    )

    assert mr_mtl_client.drift_penalty_weight == 10.0

    assert pytest.approx(mr_mtl_loss.detach().item(), abs=0.02) == (mr_mtl_client.drift_penalty_weight / 2.0) * (
        1.5 + 0.06 + 24.0 + 81.92 + 0.16 + 0.32
    )


@pytest.mark.parametrize("type,model", [(MrMtlClient, SmallCnn())])
def test_compute_loss(get_client: MrMtlClient) -> None:  # noqa
    torch.manual_seed(42)
    mr_mtl_client = get_client
    mr_mtl_client.initial_global_model = SmallCnn()
    mr_mtl_client.initial_global_model.eval()
    mr_mtl_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {}
    mr_mtl_client.criterion = torch.nn.CrossEntropyLoss()
    drift_penalty_weight = 1.0

    params = [val.cpu().numpy() for _, val in mr_mtl_client.model.state_dict().items()]
    packed_params = mr_mtl_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    mr_mtl_client.set_parameters(packed_params, config, fitting_round=True)
    mr_mtl_client.update_before_train(current_server_round=4)

    assert mr_mtl_client.drift_penalty_weight == 1.0

    # Make sure the local model is set to train
    assert mr_mtl_client.model.training is True
    for param in mr_mtl_client.model.parameters():
        assert param.requires_grad is True

    # Make sure the initial global model is not set to train
    assert mr_mtl_client.initial_global_model.training is False
    for param in mr_mtl_client.initial_global_model.parameters():
        assert param.requires_grad is False

    perturbed_params = [layer_weights + 0.1 for layer_weights in params]

    params_dict = zip(mr_mtl_client.model.state_dict().keys(), perturbed_params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    mr_mtl_client.model.load_state_dict(state_dict, strict=True)

    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    training_loss = mr_mtl_client.compute_training_loss(preds, {}, target)
    mr_mtl_client.model.eval()
    evaluation_loss = mr_mtl_client.compute_evaluation_loss(preds, {}, target)
    assert isinstance(training_loss.backward, dict)
    assert pytest.approx(54.7938, abs=0.01) == training_loss.backward["backward"].item()
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()
