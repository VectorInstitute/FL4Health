import mock
import pytest
import torch
from flwr.common import Config, Metrics
from torch.optim import Optimizer

from fl4health.clients.fedrep_client import FedRepClient, FedRepTrainMode
from fl4health.model_bases.fedrep_base import FedRepModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


@pytest.mark.parametrize("type,model", [(FedRepClient, FedRepModel(FeatureCnn(), HeadCnn()))])
def test_client_setup_and_parameter_exchange(get_client: FedRepClient) -> None:  # noqa
    torch.manual_seed(42)
    fedrep_client = get_client

    # Make sure that the train mode enum  is set to the right value
    assert fedrep_client.fedrep_train_mode == FedRepTrainMode.HEAD
    assert isinstance(fedrep_client.model, FedRepModel)

    # Set the exchanger to the fixed layer exchanger
    fedrep_client.parameter_exchanger = fedrep_client.get_parameter_exchanger({})
    assert isinstance(fedrep_client.parameter_exchanger, FixedLayerExchanger)

    # Setting the server round to 2 so we use the partial exchange.
    config: Config = {"current_server_round": 2}

    original_head_params = [val.cpu().detach().numpy() for val in fedrep_client.model.head_module.parameters()]
    modified_base_params = [val + 1.0 for val in fedrep_client.get_parameters(config)]
    fedrep_client.set_parameters(modified_base_params, config, fitting_round=True)
    base_module_parameters = list(fedrep_client.model.base_module.parameters())
    head_module_parameters = list(fedrep_client.model.head_module.parameters())

    # Tensors should be conv1 weights, biases, conv2 weights, biases (so 4 total)
    assert len(modified_base_params) == 4
    assert len(base_module_parameters) == len(modified_base_params)
    assert len(head_module_parameters) == len(original_head_params)
    # Make sure the base module weights were set correctly
    for params_1, params_2 in zip(base_module_parameters, modified_base_params):
        assert pytest.approx(torch.sum(params_1.cpu().detach() - torch.Tensor(params_2)), abs=0.0001) == 0.0
    # Make sure the head module weights were left un-touched
    for params_1, params_2 in zip(head_module_parameters, original_head_params):
        assert pytest.approx(torch.sum(params_1.cpu().detach() - torch.Tensor(params_2)), abs=0.0001) == 0.0

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(FedRepClient, FedRepModel(FeatureCnn(), HeadCnn()))])
def test_prepare_client_for_train_phase(get_client: FedRepClient) -> None:  # noqa
    torch.manual_seed(42)
    fedrep_client = get_client
    assert isinstance(fedrep_client.model, FedRepModel)
    # Make sure that the train mode enum  is set to the right value
    assert fedrep_client.fedrep_train_mode == FedRepTrainMode.HEAD
    # Make sure that none of the model components are frozen upon first creation.
    for params in fedrep_client.model.parameters():
        assert params.requires_grad
    # Prepare training call for the head module, should freeze the base layer components
    fedrep_client._prepare_train_head()
    assert fedrep_client.fedrep_train_mode == FedRepTrainMode.HEAD
    for params in fedrep_client.model.base_module.parameters():
        assert not params.requires_grad
    for params in fedrep_client.model.head_module.parameters():
        assert params.requires_grad
    # Prepare training call for the representation module, should freeze the head components
    fedrep_client._prepare_train_representations()
    assert fedrep_client.fedrep_train_mode == FedRepTrainMode.REPRESENTATION
    for params in fedrep_client.model.base_module.parameters():
        assert params.requires_grad
    for params in fedrep_client.model.head_module.parameters():
        assert not params.requires_grad

    torch.seed()  # resetting the seed at the end, just to be safe


@pytest.mark.parametrize("type,model", [(FedRepClient, FedRepModel(FeatureCnn(), HeadCnn()))])
def test_dictionary_modification_and_config_extraction(get_client: FedRepClient) -> None:  # noqa
    torch.manual_seed(42)
    fedrep_client = get_client
    loss_dict = {"loss_1": 3.5, "loss_2": 4.5}
    metrics_dict: Metrics = {"metric_1": 4.5, "metric_2": 5.5, "metric_3": 6.5}
    fedrep_client._prefix_loss_and_metrics_dictionaries("test", loss_dict, metrics_dict)
    assert {"test_loss_1", "test_loss_2"} == set(loss_dict.keys())
    assert {"test_metric_1", "test_metric_2", "test_metric_3"} == set(metrics_dict.keys())

    epoch_config: Config = {
        "current_server_round": 3,
        "evaluate_after_fit": True,
        "local_head_epochs": 5,
        "local_rep_epochs": 10,
    }
    (
        (local_head_epochs, local_rep_epochs, local_head_steps, local_rep_steps),
        current_server_round,
        evaluate_after_fit,
    ) = fedrep_client.process_fed_rep_config(epoch_config)

    assert local_head_epochs == 5
    assert local_rep_epochs == 10
    assert local_head_steps is None
    assert local_rep_steps is None
    assert current_server_round == 3
    assert evaluate_after_fit

    step_config: Config = {
        "current_server_round": 4,
        "evaluate_after_fit": False,
        "local_head_steps": 10,
        "local_rep_steps": 5,
    }
    (
        (local_head_epochs, local_rep_epochs, local_head_steps, local_rep_steps),
        current_server_round,
        evaluate_after_fit,
    ) = fedrep_client.process_fed_rep_config(step_config)

    assert local_head_steps == 10
    assert local_rep_steps == 5
    assert local_head_epochs is None
    assert local_rep_epochs is None
    assert current_server_round == 4
    assert not evaluate_after_fit

    bad_mix_config: Config = {
        "current_server_round": 5,
        "evaluate_after_fit": True,
        "local_head_epochs": 10,
        "local_rep_steps": 5,
    }
    with pytest.raises(ValueError):
        fedrep_client.process_fed_rep_config(bad_mix_config)

    bad_both_config: Config = {
        "current_server_round": 6,
        "evaluate_after_fit": False,
        "local_head_steps": 10,
        "local_rep_steps": 5,
        "local_head_epochs": 5,
        "local_rep_epochs": 10,
    }
    with pytest.raises(ValueError):
        fedrep_client.process_fed_rep_config(bad_both_config)

    torch.seed()  # resetting the seed at the end, just to be safe


def get_optimizer_patch_1(self: FedRepClient, config: Config) -> dict[str, Optimizer]:
    assert isinstance(self.model, FedRepModel)
    head_optimizer = torch.optim.AdamW(self.model.head_module.parameters(), lr=0.01)
    rep_optimizer = torch.optim.AdamW(self.model.base_module.parameters(), lr=0.01)
    return {"head": head_optimizer, "representation": rep_optimizer}


def get_optimizer_patch_2(self: FedRepClient, config: Config) -> dict[str, Optimizer]:
    assert isinstance(self.model, FedRepModel)
    head_optimizer = torch.optim.AdamW(self.model.head_module.parameters(), lr=0.01)
    rep_optimizer = torch.optim.AdamW(self.model.base_module.parameters(), lr=0.01)
    return {"test_1": head_optimizer, "test_2": rep_optimizer}


@pytest.mark.parametrize("type,model", [(FedRepClient, FedRepModel(FeatureCnn(), HeadCnn()))])
def sanity_check_optimizer_safeties(get_client: FedRepClient) -> None:  # noqa
    torch.manual_seed(42)
    with mock.patch.object(FedRepClient, "get_optimizer", new=get_optimizer_patch_1):
        fedrep_client = get_client
        fedrep_client.set_optimizer({})
        assert len(fedrep_client.optimizers) == 2

    with mock.patch.object(FedRepClient, "get_optimizer", new=get_optimizer_patch_2):
        fedrep_client = get_client
        with pytest.raises(AssertionError):
            fedrep_client.set_optimizer({})

    torch.seed()  # resetting the seed at the end, just to be safe
