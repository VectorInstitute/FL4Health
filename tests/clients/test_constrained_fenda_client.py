import copy

import numpy as np
import pytest
import torch
from flwr.common.typing import Config

from fl4health.clients.constrained_fenda_client import ConstrainedFendaClient
from fl4health.model_bases.fenda_base import FendaModelWithFeatureState
from tests.clients.fixtures import get_constrained_fenda_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_getting_parameters(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client
    # Setting the current server to 2, so that we don't set all of the weights.
    config: Config = {"current_server_round": 2}

    assert isinstance(const_fenda_client.model, FendaModelWithFeatureState)
    params_local = [
        copy.deepcopy(val.cpu().numpy())
        for val in const_fenda_client.model.first_feature_extractor.state_dict().values()
    ]
    params_global = [
        copy.deepcopy(val.cpu().numpy())
        for val in const_fenda_client.model.second_feature_extractor.state_dict().values()
    ]

    # "Do some training"
    loss = {
        "loss": 0.0,
    }
    # Do an update after training. This should be setting the required old models for the contrastive loss
    # calculations in the next round.
    const_fenda_client.update_after_train(0, loss, config)

    # Get parameters that we send to the server for aggregation. Should be the params_global values.
    global_params_for_server = const_fenda_client.get_parameters(config)
    # Perturb the parameters to mimic server-side aggregation.
    global_params_from_server = [nd_array + 1.0 for nd_array in global_params_for_server]
    # Set the parameters on the client side to mimic communication.
    const_fenda_client.set_parameters(global_params_from_server, config, True)

    # Make sure the old local module exists and is frozen
    assert isinstance(const_fenda_client.old_local_module, torch.nn.Module)
    assert const_fenda_client.old_local_module.training is False
    for param in const_fenda_client.old_local_module.parameters():
        assert param.requires_grad is False

    # Confirm that the weights from previous training are preserved in the local module
    old_local_module_params = [val.cpu().numpy() for val in const_fenda_client.old_local_module.state_dict().values()]
    for i in range(len(params_local)):
        assert (params_local[i] == old_local_module_params[i]).all()

    # Make sure the old global module exists and is frozen
    assert isinstance(const_fenda_client.old_global_module, torch.nn.Module)
    assert const_fenda_client.old_global_module.training is False
    for param in const_fenda_client.old_global_module.parameters():
        assert param.requires_grad is False

    # Confirm that the weights from previous training are preserved in the global module
    old_global_module_params = [
        val.cpu().numpy() for val in const_fenda_client.old_global_module.state_dict().values()
    ]
    for i in range(len(params_global)):
        assert (params_global[i] == old_global_module_params[i]).all()

    # Now check that the local parameters were not modified in the server communication and the global parameters were.
    new_params_local = [
        val.cpu().numpy() for val in const_fenda_client.model.first_feature_extractor.state_dict().values()
    ]
    assert len(params_local) > 0
    assert len(new_params_local) > 0
    for old_layer_global, new_layer_local in zip(params_local, new_params_local):
        np.testing.assert_allclose(old_layer_global, new_layer_local, rtol=1e-5, atol=0)

    new_params_global = [
        val.cpu().numpy() for val in const_fenda_client.model.second_feature_extractor.state_dict().values()
    ]
    assert len(global_params_from_server) > 0
    assert len(new_params_global) > 0
    for layer_from_server, new_layer_global in zip(global_params_from_server, new_params_global):
        np.testing.assert_allclose(layer_from_server, new_layer_global, rtol=0, atol=1e-5)

    # Now we do a little "training," updating the model weights and make sure the old module weights do not change.
    for weights in const_fenda_client.model.state_dict().values():
        weights += 0.1

    # old_local_module_params should differ from the "trained" local module weights by 0.1
    old_local_module_params = [val.cpu().numpy() for val in const_fenda_client.old_local_module.state_dict().values()]
    trained_local_module_params = [
        val.cpu().numpy() for val in const_fenda_client.model.first_feature_extractor.state_dict().values()
    ]
    for i in range(len(trained_local_module_params)):
        np.testing.assert_allclose(
            (trained_local_module_params[i] - 0.1), old_local_module_params[i], rtol=0, atol=1e-5
        )

    # old_global_module_params should differ from the "trained" global module weights by 1.1 (training and
    # aggregation perturbations)
    old_global_module_params = [
        val.cpu().numpy() for val in const_fenda_client.old_global_module.state_dict().values()
    ]
    trained_global_module_params = [
        val.cpu().numpy() for val in const_fenda_client.model.second_feature_extractor.state_dict().values()
    ]
    for i in range(len(trained_global_module_params)):
        np.testing.assert_allclose(
            (trained_global_module_params[i] - 1.1), old_global_module_params[i], rtol=0, atol=1e-5
        )

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_setting_global_model(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client

    assert const_fenda_client.initial_global_module is None
    assert isinstance(const_fenda_client.model, FendaModelWithFeatureState)

    global_params = [
        copy.deepcopy(val.cpu().numpy())
        for _, val in const_fenda_client.model.second_feature_extractor.state_dict().items()
    ]

    const_fenda_client.update_before_train(0)
    # Because a PerFCL loss has been set we should save the initial_global_module in update_before_train
    assert const_fenda_client.initial_global_module is not None

    aggregate_params = [
        copy.deepcopy(val.cpu().numpy()) for _, val in const_fenda_client.initial_global_module.state_dict().items()
    ]
    # Make sure the fenda aggregated module parameters are equal to the global module parameters
    for i in range(len(aggregate_params)):
        assert (aggregate_params[i] == global_params[i]).all()

    # Make sure the aggregated module is not set to train
    assert const_fenda_client.initial_global_module.training is False
    for param in const_fenda_client.initial_global_module.parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert const_fenda_client.model.training is True
    for param in const_fenda_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_not_setting_global_model(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client
    # Explicitly set the initial global module to None for the client.
    const_fenda_client.loss_container.perfcl_loss_config = None

    assert isinstance(const_fenda_client.model, FendaModelWithFeatureState)
    assert const_fenda_client.initial_global_module is None

    const_fenda_client.update_before_train(0)
    # Because a PerFCL loss has NOT been set we should NOT save the initial_global_module in update_before_train
    assert const_fenda_client.initial_global_module is None

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_setting_old_models(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client

    assert const_fenda_client.old_local_module is None
    assert const_fenda_client.old_global_module is None
    assert isinstance(const_fenda_client.model, FendaModelWithFeatureState)

    local_params = [
        copy.deepcopy(val.cpu().numpy())
        for _, val in const_fenda_client.model.first_feature_extractor.state_dict().items()
    ]
    global_params = [
        copy.deepcopy(val.cpu().numpy())
        for _, val in const_fenda_client.model.second_feature_extractor.state_dict().items()
    ]
    loss = {
        "loss": 0.0,
    }
    const_fenda_client.update_after_train(0, loss, {})

    assert const_fenda_client.old_local_module is not None
    old_local_params = [
        copy.deepcopy(val.cpu().numpy()) for _, val in const_fenda_client.old_local_module.state_dict().items()
    ]

    assert const_fenda_client.old_global_module is not None
    old_global_params = [
        copy.deepcopy(val.cpu().numpy()) for _, val in const_fenda_client.old_global_module.state_dict().items()
    ]

    # Make sure the FENDA old local module parameters are equal to the local module parameters
    for i in range(len(local_params)):
        assert (local_params[i] == old_local_params[i]).all()

    # Make sure the FENDA old global module parameters are equal to the global module parameters
    for i in range(len(global_params)):
        assert (global_params[i] == old_global_params[i]).all()

    # Make sure the old global and local module is not set to train
    assert const_fenda_client.old_local_module.training is False
    for param in const_fenda_client.old_local_module.parameters():
        assert param.requires_grad is False
    assert const_fenda_client.old_global_module.training is False
    for param in const_fenda_client.old_global_module.parameters():
        assert param.requires_grad is False

    # Make sure the original model is still set to train
    assert const_fenda_client.model.training is True
    for param in const_fenda_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_setting_not_setting_old_models(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client
    # Setting both loss configurations to None so that we don't bother saving the old local and global modules.
    const_fenda_client.loss_container.contrastive_loss_config = None
    const_fenda_client.loss_container.perfcl_loss_config = None

    assert const_fenda_client.old_local_module is None
    assert const_fenda_client.old_global_module is None
    assert isinstance(const_fenda_client.model, FendaModelWithFeatureState)

    loss = {
        "loss": 0.0,
    }
    const_fenda_client.update_after_train(0, loss, {})

    # Make sure they are still None
    assert const_fenda_client.old_local_module is None
    assert const_fenda_client.old_global_module is None

    # Make sure the original model is still set to train
    assert const_fenda_client.model.training is True
    for param in const_fenda_client.model.parameters():
        assert param.requires_grad is True

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_computing_loss(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    torch.manual_seed(42)
    const_fenda_client = get_constrained_fenda_client
    const_fenda_client.criterion = torch.nn.CrossEntropyLoss()

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

    training_loss = const_fenda_client.compute_training_loss(preds=preds, target=target, features=features)
    evaluation_loss = const_fenda_client.compute_evaluation_loss(preds=preds, target=target, features=features)
    assert isinstance(training_loss.backward["backward"], torch.Tensor)
    assert pytest.approx(0.8132616, abs=0.0001) == evaluation_loss.checkpoint.item()
    assert pytest.approx(6.1940451, abs=0.0001) == training_loss.backward["backward"].item()
    assert evaluation_loss.checkpoint.item() != training_loss.backward["backward"].item()
    assert training_loss.additional_losses == evaluation_loss.additional_losses
    assert training_loss.additional_losses["loss"] == evaluation_loss.checkpoint.item()
    assert training_loss.additional_losses["total_loss"] == training_loss.backward["backward"].item()

    auxiliary_loss_total = (training_loss.backward["backward"] - evaluation_loss.checkpoint).item()
    cosine_similarity_loss = training_loss.additional_losses["cos_sim_loss"].item()
    vanilla_contrastive_loss = training_loss.additional_losses["contrastive_loss"].item()
    global_feature_contrastive_loss = training_loss.additional_losses["global_feature_contrastive_loss"].item()
    local_feature_contrastive_loss = training_loss.additional_losses["local_feature_contrastive_loss"].item()
    assert pytest.approx(auxiliary_loss_total, abs=0.001) == (
        global_feature_contrastive_loss
        + local_feature_contrastive_loss
        + vanilla_contrastive_loss
        + cosine_similarity_loss
    )
    assert pytest.approx(cosine_similarity_loss, abs=0.001) == 1.0
    assert pytest.approx(vanilla_contrastive_loss, abs=0.001) == 2.1269
    assert pytest.approx(global_feature_contrastive_loss, abs=0.001) == 0.1269
    assert pytest.approx(local_feature_contrastive_loss, abs=0.001) == 2.1269

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_feature_flatten(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    const_fenda_client = get_constrained_fenda_client
    features = torch.rand((8, 2, 3, 4))
    flattened_features = const_fenda_client._flatten(features)
    assert features.shape == (8, 2, 3, 4)
    assert flattened_features.shape == (8, 24)


@pytest.mark.parametrize("local_module,global_module,head_module", [(FeatureCnn(), FeatureCnn(), FendaHeadCnn())])
def test_perfcl_keys_present(get_constrained_fenda_client: ConstrainedFendaClient) -> None:  # noqa
    const_fenda_client = get_constrained_fenda_client
    keys_present = {
        "old_local_features": torch.Tensor([1, 2, 3]),
        "old_global_features": torch.Tensor([1, 2, 3]),
        "initial_global_features": torch.Tensor([1, 2, 3]),
        "one_more_key": torch.Tensor([1, 2, 3]),
    }
    keys_not_present = {
        "old_global_features": torch.Tensor([1, 2, 3]),
        "initial_global_features": torch.Tensor([1, 2, 3]),
    }
    assert const_fenda_client._perfcl_keys_present(keys_present)
    assert not const_fenda_client._perfcl_keys_present(keys_not_present)
