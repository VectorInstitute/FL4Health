import pytest
import torch
from flwr.common import Config

from fl4health.clients.fenda_ditto_client import FendaDittoClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn, HeadCnn, SmallCnn


@pytest.mark.parametrize("type,model", [(FendaDittoClient, FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn()))])
def test_setting_initial_weights(get_client: FendaDittoClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_ditto_client = get_client
    fenda_ditto_client.global_model = SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn())
    fenda_ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {"current_server_round": 1}
    drift_penalty_weight = 10.0

    params = [val.cpu().numpy() + 1.0 for _, val in fenda_ditto_client.global_model.state_dict().items()]
    packed_params = fenda_ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    fenda_ditto_client.set_parameters(packed_params, config, fitting_round=True)
    fenda_ditto_client.update_before_train(1)

    global_model_params = [val.detach().clone() for _, val in fenda_ditto_client.global_model.state_dict().items()]

    fenda_model_params = [
        val.detach().clone()
        for name, val in fenda_ditto_client.model.state_dict().items()
        if name.startswith("second_feature_extractor.")
    ]

    # First fitting round we should set both the global and local models to params and store the global model values
    assert fenda_ditto_client.drift_penalty_tensors is not None
    # Tensors should be conv1 weights, biases, conv2 weights, biases (so 4 total)
    assert len(fenda_ditto_client.drift_penalty_tensors) == 4
    # Make sure that we saved the right parameters
    for layer_init_global_tensor, layer_params in zip(fenda_ditto_client.drift_penalty_tensors, params):
        assert pytest.approx(torch.sum(layer_init_global_tensor - layer_params), abs=0.0001) == 0.0
    # Make sure the global model was set correctly
    for global_model_layer_params, layer_params in zip(global_model_params, params):
        assert pytest.approx(torch.sum(global_model_layer_params - layer_params), abs=0.0001) == 0.0
    # Make sure the local model was set correctly
    for local_model_layer_params, layer_params in zip(fenda_model_params, params):
        assert pytest.approx(torch.sum(local_model_layer_params - layer_params), abs=0.0001) == 0.0

    # Now we update in a later round
    config = {"current_server_round": 2}
    params = [param + 1.0 for param in params]
    packed_params = fenda_ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    fenda_ditto_client.set_parameters(packed_params, config, fitting_round=True)
    fenda_ditto_client.update_before_train(2)
    global_model_params = [val.detach().clone() for _, val in fenda_ditto_client.global_model.state_dict().items()]

    fenda_model_params = [
        val.detach().clone()
        for name, val in fenda_ditto_client.model.state_dict().items()
        if name.startswith("second_feature_extractor.")
    ]
    # Make sure that we saved the right parameters
    for layer_init_global_tensor, layer_params in zip(fenda_ditto_client.drift_penalty_tensors, params):
        assert pytest.approx(torch.sum(layer_init_global_tensor - layer_params), abs=0.0001) == 0.0
    # Make sure the global model was set correctly
    for global_model_layer_params, layer_params in zip(global_model_params, params):
        assert pytest.approx(torch.sum(global_model_layer_params - layer_params), abs=0.0001) == 0.0
    # Make sure the local model was set correctly
    for local_model_layer_params, layer_params in zip(fenda_model_params, params):
        assert pytest.approx(torch.sum(local_model_layer_params - layer_params), abs=0.0001) == 0.0


@pytest.mark.parametrize("type,model", [(FendaDittoClient, FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn()))])
def test_forming_ditto_loss(get_client: FendaDittoClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_ditto_client = get_client
    fenda_ditto_client.global_model = SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn())
    fenda_ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    drift_penalty_weight = 1.0
    config: Config = {"current_server_round": 2}

    # feature extractor is given to FENDA model
    fenda_ditto_client.model.first_feature_extractor.load_state_dict(
        fenda_ditto_client.global_model.base_module.state_dict()
    )

    params = [val.cpu().numpy() + 0.1 for _, val in fenda_ditto_client.global_model.state_dict().items()]
    packed_params = fenda_ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    fenda_ditto_client.set_parameters(packed_params, config, fitting_round=True)
    fenda_ditto_client.update_before_train(4)

    ditto_loss = fenda_ditto_client.penalty_loss_function(
        fenda_ditto_client.model.first_feature_extractor,
        fenda_ditto_client.drift_penalty_tensors,
        fenda_ditto_client.drift_penalty_weight,
    )

    assert fenda_ditto_client.drift_penalty_weight == 1.0
    assert pytest.approx(ditto_loss.detach().item(), abs=0.02) == (fenda_ditto_client.drift_penalty_weight / 2.0) * (
        1.5 + 0.06 + 24 + 0.16
    )


@pytest.mark.parametrize("type,model", [(FendaDittoClient, FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn()))])
def test_compute_loss(get_client: FendaDittoClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_ditto_client = get_client
    fenda_ditto_client.global_model = SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn())
    fenda_ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {"current_server_round": 2}
    fenda_ditto_client.criterion = torch.nn.CrossEntropyLoss()
    drift_penalty_weight = 1.0

    params = [val.cpu().numpy() for _, val in fenda_ditto_client.global_model.state_dict().items()]
    packed_params = fenda_ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    fenda_ditto_client.set_parameters(packed_params, config, fitting_round=True)
    fenda_ditto_client.update_before_train(4)

    # Training: both FENDA models are updated
    state_dict = fenda_ditto_client.global_model.base_module.state_dict()
    for key in state_dict:
        state_dict[key] += 0.1
    fenda_ditto_client.model.first_feature_extractor.load_state_dict(state_dict)
    fenda_ditto_client.model.second_feature_extractor.load_state_dict(state_dict)

    preds = {"global": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "local": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    training_loss = fenda_ditto_client.compute_training_loss(preds, {}, target)
    fenda_ditto_client.global_model.eval()
    fenda_ditto_client.model.eval()
    evaluation_loss = fenda_ditto_client.compute_evaluation_loss(preds, {}, target)

    assert isinstance(training_loss.backward, dict)
    assert pytest.approx(13.673, abs=0.01) == training_loss.backward["backward"].item()
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()


@pytest.mark.parametrize("type,model", [(FendaDittoClient, FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn()))])
def test_compute_loss_freeze_global_feature_extractor(get_client: FendaDittoClient) -> None:  # noqa
    torch.manual_seed(42)
    fenda_ditto_client = get_client
    fenda_ditto_client.global_model = SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn())
    fenda_ditto_client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    config: Config = {"current_server_round": 2}
    fenda_ditto_client.criterion = torch.nn.CrossEntropyLoss()
    drift_penalty_weight = 1.0
    fenda_ditto_client.freeze_global_feature_extractor = True

    params = [val.cpu().numpy() for _, val in fenda_ditto_client.global_model.state_dict().items()]
    packed_params = fenda_ditto_client.parameter_exchanger.pack_parameters(params, drift_penalty_weight)
    fenda_ditto_client.set_parameters(packed_params, config, fitting_round=True)
    fenda_ditto_client.update_before_train(4)

    # Training: only local FENDA model is updated
    state_dict = fenda_ditto_client.global_model.base_module.state_dict()
    for key in state_dict:
        state_dict[key] += 0.1
    fenda_ditto_client.model.first_feature_extractor.load_state_dict(state_dict)

    preds = {"global": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "local": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    training_loss = fenda_ditto_client.compute_training_loss(preds, {}, target)
    fenda_ditto_client.global_model.eval()
    fenda_ditto_client.model.eval()
    evaluation_loss = fenda_ditto_client.compute_evaluation_loss(preds, {}, target)

    assert isinstance(training_loss.backward, dict)
    assert pytest.approx(13.673, abs=0.01) == training_loss.backward["backward"].item()
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()


@pytest.mark.parametrize("type,model", [(FendaDittoClient, FendaModel(SmallCnn(), FeatureCnn(), FendaHeadCnn()))])
def test_setup_client_with_incorrect_model(get_client: FendaDittoClient) -> None:  # noqa
    fenda_ditto_client = get_client
    fenda_ditto_client.global_model = SequentiallySplitExchangeBaseModel(SmallCnn(), HeadCnn())
    # Should raise an assertion error because the model type is incorrect.
    with pytest.raises(AssertionError):
        fenda_ditto_client._check_shape_match()
