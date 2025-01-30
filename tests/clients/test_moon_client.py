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
    config: Config = {"current_server_round": 1}

    params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    new_params = [layer_weights + 0.1 for layer_weights in params]
    moon_client.set_parameters(new_params, config, fitting_round=True)

    # Make sure the MOON model parameters are equal to the global model parameters
    new_moon_model_params = [val.cpu().numpy() for val in moon_client.model.state_dict().values()]

    for i in range(len(params)):
        assert (new_params[i] == new_moon_model_params[i]).all()

    # Make sure the old model list is still empty
    assert len(moon_client.old_models_list) == 0

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_setting_global_model(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client

    assert moon_client.global_model is None

    params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    moon_client.update_before_train(0)

    assert moon_client.global_model is not None

    global_params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.global_model.state_dict().values()]
    # Make sure the MOON model parameters are equal to the global model parameters
    for i in range(len(params)):
        assert (params[i] == global_params[i]).all()

    # Make sure the global model is not set to train
    assert moon_client.global_model.training is False
    for param in moon_client.global_model.parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert moon_client.model.training is True
    for param in moon_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_setting_old_models(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client

    assert len(moon_client.old_models_list) == 0

    params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    loss = {
        "loss": 0.0,
    }
    moon_client.update_after_train(0, loss, {})

    # Assert we stored the old model
    assert len(moon_client.old_models_list) == 1

    old_model_params = [
        copy.deepcopy(val.cpu().numpy()) for val in moon_client.old_models_list[0].state_dict().values()
    ]
    # Make sure the MOON model parameters are equal to the old model parameters
    for i in range(len(params)):
        assert (params[i] == old_model_params[i]).all()

    # Make sure the all old models is not set to train
    assert moon_client.old_models_list[0].training is False
    for param in moon_client.old_models_list[0].parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert moon_client.model.training is True
    for param in moon_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_getting_parameters(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    config: Config = {"current_server_round": 1}

    assert len(moon_client.old_models_list) == 0

    params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    loss = {
        "loss": 0.0,
    }
    moon_client.update_after_train(0, loss, config)
    # Mocking sending parameters to the server, need to make sure the old_model_list is updated
    _ = moon_client.get_parameters(config)
    new_params = [layer_weights + 0.1 for layer_weights in params]
    # Setting parameters once to represent an evaluation set parameters
    moon_client.set_parameters(new_params, config, fitting_round=False)
    # Setting parameters again to represent a training set parameters
    moon_client.set_parameters(new_params, config, fitting_round=True)
    moon_client.update_before_train(0)

    # Assert we stored the old model and the global model
    assert len(moon_client.old_models_list) == 1
    assert moon_client.global_model is not None

    old_model_params = [val.cpu().numpy() for val in moon_client.old_models_list[-1].state_dict().values()]
    global_model_params = [val.cpu().numpy() for val in moon_client.global_model.state_dict().values()]
    # Make sure the MOON model parameters are equal to the global model parameters
    new_moon_model_params = [val.cpu().numpy() for val in moon_client.model.state_dict().values()]

    for i in range(len(params)):
        assert (params[i] == old_model_params[i]).all()
        assert (new_params[i] == global_model_params[i]).all()
        assert (new_params[i] == new_moon_model_params[i]).all()

    # Do another round to make sure old model list doesn't expand and it contains new parameters
    # Mocking sending parameters to the server, need to make sure the old_model_list is updated
    config["current_server_round"] = 2
    moon_client.update_after_train(0, loss, config)
    _ = moon_client.get_parameters(config)
    new_params_2 = [layer_weights + 0.1 for layer_weights in params]
    # Setting parameters once to represent an evaluation set parameters
    moon_client.set_parameters(new_params, config, fitting_round=False)
    # Setting parameters again to represent a training set parameters
    moon_client.set_parameters(new_params, config, fitting_round=True)
    moon_client.update_before_train(0)

    # Assert we stored the old model
    assert len(moon_client.old_models_list) == 1

    old_model_params = [val.cpu().numpy() for val in moon_client.old_models_list[-1].state_dict().values()]
    global_model_params = [val.cpu().numpy() for val in moon_client.global_model.state_dict().values()]
    # Make sure the MOON model parameters are equal to the global model parameters
    new_moon_model_params = [val.cpu().numpy() for val in moon_client.model.state_dict().values()]

    for i in range(len(params)):
        assert (new_params[i] == old_model_params[i]).all()
        assert (new_params_2[i] == global_model_params[i]).all()
        assert (new_params_2[i] == new_moon_model_params[i]).all()

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_computing_loss(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    # Dummy to ensure the compute loss function in the moon client is executed
    moon_client.old_models_list = [moon_client.model]
    moon_client.contrastive_weight = 2.0
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
    expected_loss = 0.8132616
    expected_total_loss = 2 * 0.837868 + 0.8132616
    expected_contrastive_loss = 0.837868

    total_loss, additional_losses = moon_client.compute_loss_and_additional_losses(preds, features, target)
    assert isinstance(total_loss, torch.Tensor)
    assert pytest.approx(expected_total_loss, abs=0.0001) == total_loss.item()
    assert pytest.approx(expected_contrastive_loss, abs=0.0001) == additional_losses["contrastive_loss"]
    assert pytest.approx(expected_loss, abs=0.0001) == additional_losses["loss"].item()
    assert pytest.approx(expected_total_loss, abs=0.0001) == additional_losses["total_loss"].item()

    # Make sure the model is set to train
    moon_client.model.train()
    training_loss = moon_client.compute_training_loss(preds=preds, target=target, features=features)
    # Make sure the model is set to eval
    moon_client.model.eval()
    evaluation_loss = moon_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(expected_loss, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(expected_total_loss, abs=0.0001) == training_loss.backward["backward"].item()
    assert training_loss.additional_losses == evaluation_loss.additional_losses

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_computing_first_loss(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    # Dummy to ensure the compute loss with a blank model list is computed correctly
    moon_client.old_models_list = []
    moon_client.contrastive_weight = 2.0
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
    expected_loss = 0.8132616

    # Make sure the model is set to train
    moon_client.model.train()
    training_loss = moon_client.compute_training_loss(preds=preds, target=target, features=features)
    # Make sure the model is set to eval
    moon_client.model.eval()
    evaluation_loss = moon_client.compute_evaluation_loss(preds=preds, target=target, features=features)

    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(expected_loss, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(expected_loss, abs=0.0001) == training_loss.backward["backward"].item()
    assert len(evaluation_loss.additional_losses) == 0
    assert len(training_loss.additional_losses) == 0

    # Now lets set a local_model list entry and make sure everything comes out right.
    # Dummy to ensure the compute loss function in the moon client is executed
    moon_client.old_models_list = [moon_client.model]
    expected_total_loss = 2 * 0.837868 + 0.8132616
    expected_contrastive_loss = 0.837868

    total_loss, additional_losses = moon_client.compute_loss_and_additional_losses(preds, features, target)
    assert isinstance(total_loss, torch.Tensor)
    assert pytest.approx(expected_total_loss, abs=0.0001) == total_loss.item()
    assert pytest.approx(expected_contrastive_loss, abs=0.0001) == additional_losses["contrastive_loss"]
    assert pytest.approx(expected_loss, abs=0.0001) == additional_losses["loss"].item()
    assert pytest.approx(expected_total_loss, abs=0.0001) == additional_losses["total_loss"].item()

    # Make sure the model is set to train
    moon_client.model.train()
    training_loss = moon_client.compute_training_loss(preds=preds, target=target, features=features)
    # Make sure the model is set to eval
    moon_client.model.eval()
    evaluation_loss = moon_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(expected_loss, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(expected_total_loss, abs=0.0001) == training_loss.backward["backward"].item()
    assert training_loss.additional_losses == evaluation_loss.additional_losses

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(MoonClient, MoonModel(FeatureCnn(), HeadCnn()))])
def test_popping_old_models_when_queue_fills(get_client: MoonClient) -> None:  # noqa
    torch.manual_seed(42)
    moon_client = get_client
    moon_client.len_old_models_buffer = 2
    assert len(moon_client.old_models_list) == 0
    assert isinstance(moon_client.model, MoonModel)

    original_base_module_params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    assert len(original_base_module_params) > 0

    moon_client.update_after_train(0, {}, {})
    moon_client.update_after_train(1, {}, {})
    assert len(moon_client.old_models_list) == 2

    # Now we do a little "training," updating the model weights
    for weights in moon_client.model.state_dict().values():
        weights += 0.1

    new_base_module_params = [copy.deepcopy(val.cpu().numpy()) for val in moon_client.model.state_dict().values()]
    assert len(new_base_module_params) > 0

    moon_client.update_after_train(2, {}, {})
    assert len(moon_client.old_models_list) == 2

    # The first "old model" should have the original model parameters.
    for old_model_params, original_params in zip(
        moon_client.old_models_list[0].state_dict().values(), original_base_module_params
    ):
        assert (old_model_params.cpu().numpy() == original_params).all()

    # The second "old model" should have the new model parameters
    for old_model_params, new_params in zip(
        moon_client.old_models_list[1].state_dict().values(), new_base_module_params
    ):
        assert (old_model_params.cpu().numpy() == new_params).all()

    torch.seed()  # resetting the seed at the end, just to be safe
