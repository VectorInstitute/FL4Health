import pytest

from fl4health.clients.fedper_client import FedPerClient
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel, SequentiallySplitModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


@pytest.mark.parametrize("type,model", [(FedPerClient, SequentiallySplitModel(FeatureCnn(), HeadCnn()))])
def test_get_parameter_exchanger_with_incorrect_model(get_client: FedPerClient) -> None:  # noqa
    fedper_client = get_client
    # Should raise an assertion error because the model type is incorrect.
    with pytest.raises(AssertionError):
        fedper_client.get_parameter_exchanger({})


@pytest.mark.parametrize("type,model", [(FedPerClient, SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn()))])
def test_get_parameter_exchanger_with_correct_model(get_client: FedPerClient) -> None:  # noqa
    fedper_client = get_client
    parameter_exchanger = fedper_client.get_parameter_exchanger({})
    assert isinstance(parameter_exchanger, FixedLayerExchanger)
    target_layers_to_transfer = [
        "base_module.conv1.weight",
        "base_module.conv1.bias",
        "base_module.conv2.weight",
        "base_module.conv2.bias",
    ]
    exchanger_layers_to_transfer = parameter_exchanger.layers_to_transfer
    assert len(target_layers_to_transfer) == len(exchanger_layers_to_transfer)
    assert set(target_layers_to_transfer) == set(exchanger_layers_to_transfer)
