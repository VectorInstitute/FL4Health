import copy

import numpy as np
import pytest
import torch
from flwr.common import Config

from fl4health.clients.perfcl_client import PerFclClient
from fl4health.model_bases.perfcl_base import PerFclModel
from tests.clients.fixtures import get_perfcl_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_getting_parameters(get_perfcl_client: PerFclClient) -> None:  # noqa
    torch.manual_seed(42)
    perfcl_client = get_perfcl_client
    # Setting the current server to 2, so that we don't set all of the weights.
    config: Config = {"current_server_round": 2}

    assert isinstance(perfcl_client.model, PerFclModel)
    params_local = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.model.first_feature_extractor.state_dict().values()
    ]
    params_global = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.model.second_feature_extractor.state_dict().values()
    ]

    # "Do some training"
    loss = {
        "loss": 0.0,
    }
    # Do an update after training. This should be setting the required old models for the contrastive loss
    # calculations in the next round.
    perfcl_client.update_after_train(0, loss, config)

    # Get parameters that we send to the server for aggregation. Should be the params_global values.
    global_params_for_server = perfcl_client.get_parameters(config)
    # Perturb the parameters to mimic server-side aggregation.
    global_params_from_server = [nd_array + 1.0 for nd_array in global_params_for_server]
    # Set the parameters on the client side to mimic communication.
    perfcl_client.set_parameters(global_params_from_server, config, True)

    # Make sure the old local module exists and is frozen
    assert isinstance(perfcl_client.old_local_module, torch.nn.Module)
    assert perfcl_client.old_local_module.training is False
    for param in perfcl_client.old_local_module.parameters():
        assert param.requires_grad is False

    # Confirm that the weights from previous training are preserved in the local module
    old_local_module_params = [val.cpu().numpy() for val in perfcl_client.old_local_module.state_dict().values()]
    for i in range(len(params_local)):
        assert (params_local[i] == old_local_module_params[i]).all()

    # Make sure the old global module exists and is frozen
    assert isinstance(perfcl_client.old_global_module, torch.nn.Module)
    assert perfcl_client.old_global_module.training is False
    for param in perfcl_client.old_global_module.parameters():
        assert param.requires_grad is False

    # Confirm that the weights from previous training are preserved in the global module
    old_global_module_params = [val.cpu().numpy() for val in perfcl_client.old_global_module.state_dict().values()]
    for i in range(len(params_global)):
        assert (params_global[i] == old_global_module_params[i]).all()

    # Now check that the local parameters were not modified in the server communication and the global parameters were.
    new_params_local = [val.cpu().numpy() for val in perfcl_client.model.first_feature_extractor.state_dict().values()]
    assert len(params_local) > 0
    assert len(new_params_local) > 0
    for old_layer_global, new_layer_local in zip(params_local, new_params_local):
        np.testing.assert_allclose(old_layer_global, new_layer_local, rtol=1e-5, atol=0)

    new_params_global = [
        val.cpu().numpy() for val in perfcl_client.model.second_feature_extractor.state_dict().values()
    ]
    assert len(global_params_from_server) > 0
    assert len(new_params_global) > 0
    for layer_from_server, new_layer_global in zip(global_params_from_server, new_params_global):
        np.testing.assert_allclose(layer_from_server, new_layer_global, rtol=0, atol=1e-5)

    # Now we do a little "training," updating the model weights and make sure the old module weights do not change.
    for weights in perfcl_client.model.state_dict().values():
        weights += 0.1

    # old_local_module_params should differ from the "trained" local module weights by 0.1
    old_local_module_params = [val.cpu().numpy() for val in perfcl_client.old_local_module.state_dict().values()]
    trained_local_module_params = [
        val.cpu().numpy() for val in perfcl_client.model.first_feature_extractor.state_dict().values()
    ]
    for i in range(len(trained_local_module_params)):
        np.testing.assert_allclose(
            (trained_local_module_params[i] - 0.1), old_local_module_params[i], rtol=0, atol=1e-5
        )

    # old_global_module_params should differ from the "trained" global module weights by 1.1 (training and
    # aggregation perturbations)
    old_global_module_params = [val.cpu().numpy() for val in perfcl_client.old_global_module.state_dict().values()]
    trained_global_module_params = [
        val.cpu().numpy() for val in perfcl_client.model.second_feature_extractor.state_dict().values()
    ]
    for i in range(len(trained_global_module_params)):
        np.testing.assert_allclose(
            (trained_global_module_params[i] - 1.1), old_global_module_params[i], rtol=0, atol=1e-5
        )

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_setting_initial_global_module(get_perfcl_client: PerFclClient) -> None:  # noqa
    torch.manual_seed(42)
    perfcl_client = get_perfcl_client

    # When we're doing the first round of training, the initial global module should be None until update_before_train
    # is called
    assert perfcl_client.initial_global_module is None
    assert isinstance(perfcl_client.model, PerFclModel)

    global_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.model.second_feature_extractor.state_dict().values()
    ]
    perfcl_client.update_before_train(0)
    assert perfcl_client.initial_global_module is not None

    aggregate_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.initial_global_module.state_dict().values()
    ]

    # Make sure the fenda aggregated module parameters are equal to the global module parameters
    for i in range(len(aggregate_params)):
        assert (aggregate_params[i] == global_params[i]).all()

    # Make sure the aggregated module is not set to train
    assert perfcl_client.initial_global_module.training is False
    for param in perfcl_client.initial_global_module.parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert perfcl_client.model.training is True
    for param in perfcl_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_setting_old_models(get_perfcl_client: PerFclClient) -> None:  # noqa
    torch.manual_seed(42)
    perfcl_client = get_perfcl_client

    assert perfcl_client.old_local_module is None
    assert perfcl_client.old_global_module is None
    assert isinstance(perfcl_client.model, PerFclModel)

    local_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.model.first_feature_extractor.state_dict().values()
    ]
    global_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.model.second_feature_extractor.state_dict().values()
    ]
    loss = {
        "loss": 0.0,
    }
    perfcl_client.update_after_train(0, loss, {})

    assert perfcl_client.old_local_module is not None
    old_local_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.old_local_module.state_dict().values()
    ]

    assert perfcl_client.old_global_module is not None
    old_global_params = [
        copy.deepcopy(val.cpu().numpy()) for val in perfcl_client.old_global_module.state_dict().values()
    ]

    # Make sure the old local module parameters are equal to the local module parameters
    for i in range(len(local_params)):
        assert (local_params[i] == old_local_params[i]).all()

    # Make sure the old global module parameters are equal to the global module parameters
    for i in range(len(global_params)):
        assert (global_params[i] == old_global_params[i]).all()

    # Make sure the old global and local module is not set to train
    assert perfcl_client.old_local_module.training is False
    for param in perfcl_client.old_local_module.parameters():
        assert param.requires_grad is False
    assert perfcl_client.old_global_module.training is False
    for param in perfcl_client.old_global_module.parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert perfcl_client.model.training is True
    for param in perfcl_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_computing_loss(get_perfcl_client: PerFclClient) -> None:  # noqa
    torch.manual_seed(42)
    perfcl_client = get_perfcl_client
    perfcl_client.criterion = torch.nn.CrossEntropyLoss()

    local_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    old_local_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    old_global_features = torch.tensor([[0, 0, 0], [0, 0, 0]]).float()
    initial_global_features = torch.tensor([[1, 1, 1], [1, 1, 1]]).float()
    preds = {"prediction": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    features = {
        "local_features": local_features,
        "old_local_features": old_local_features,
        "global_features": global_features,
        "old_global_features": old_global_features,
        "initial_global_features": initial_global_features,
    }

    # In this case, we have not set the proper modules in the client to produce these features. So we should only
    # be computing the vanilla loss for both the training and evaluation stages
    training_loss = perfcl_client.compute_training_loss(preds=preds, target=target, features=features)
    evaluation_loss = perfcl_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(0.8132616, abs=0.0001) == training_loss.backward["backward"].item()
    assert len(evaluation_loss.additional_losses) == 1
    assert len(training_loss.additional_losses) == 1

    # Now we mock having "set" the right components. So we should compute the full set of loss components
    perfcl_client.update_before_train(0)
    perfcl_client.update_after_train(0, {}, {})

    training_loss = perfcl_client.compute_training_loss(preds=preds, target=target, features=features)
    evaluation_loss = perfcl_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(3.0671176, abs=0.0001) == training_loss.backward["backward"].item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()
    assert training_loss.additional_losses["loss"] == evaluation_loss.checkpoint.item()
    assert training_loss.additional_losses["total_loss"] == training_loss.backward["backward"].item()

    auxiliary_loss_total = (training_loss.backward["backward"] - evaluation_loss.checkpoint).item()
    contrastive_minimize = training_loss.additional_losses["global_feature_contrastive_loss"].item()
    contrastive_maximize = training_loss.additional_losses["local_feature_contrastive_loss"].item()
    assert pytest.approx(auxiliary_loss_total, abs=0.001) == (contrastive_minimize + contrastive_maximize)

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_feature_flatten(get_perfcl_client: PerFclClient) -> None:  # noqa
    perfcl_client = get_perfcl_client
    features = torch.rand((8, 2, 3, 4))
    flattened_features = perfcl_client._flatten(features)
    assert features.shape == (8, 2, 3, 4)
    assert flattened_features.shape == (8, 24)
