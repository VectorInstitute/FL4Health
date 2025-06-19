import copy

import numpy as np
import pytest
import torch
from flwr.common import Config

from fl4health.clients.fedpm_client import FedPmClient
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames
from tests.clients.fixtures import get_fedpm_client  # noqa
from tests.test_utils.models_for_test import CompositeConvNet


@pytest.mark.parametrize("model", [CompositeConvNet()])
def test_getting_parameters(get_fedpm_client: FedPmClient) -> None:  # noqa
    torch.manual_seed(42)
    fedpm_client = get_fedpm_client
    parameter_packer = ParameterPackerWithLayerNames()
    # Setting the current server to 2, so that we don't set all of the weights.
    config: Config = {"current_server_round": 2}
    assert isinstance(fedpm_client.parameter_exchanger, FedPmExchanger)

    # Save the original local module parameters before exchange
    old_params_local = {
        param_name: copy.deepcopy(param_val.cpu().numpy())
        for param_name, param_val in fedpm_client.model.state_dict().items()
    }

    # Get parameters to be sent to the server for aggregation.
    masks_for_server = fedpm_client.get_parameters(config)

    assert isinstance(fedpm_client.parameter_exchanger, FedPmExchanger)
    masks, score_param_names = parameter_packer.unpack_parameters(masks_for_server)
    assert len(score_param_names) == len(masks) == 22

    # Mimic server-side aggregation.
    aggregated_masks = []
    for mask in masks:
        aggregated_masks.append((mask + 1) / 2)
    aggregation_result = parameter_packer.pack_parameters(aggregated_masks, score_param_names)

    # Set the parameters on the client side to mimic communication.
    fedpm_client.set_parameters(aggregation_result, config, fitting_round=True)

    # Check that only the score tensors are modified and the other parameters
    # are not.
    new_params_local = {
        param_name: copy.deepcopy(param_val.cpu().numpy())
        for param_name, param_val in fedpm_client.model.state_dict().items()
    }

    for param_name, param_val in new_params_local.items():
        if "score" in param_name:
            assert not (param_val == old_params_local[param_name]).all()
        else:
            np.testing.assert_allclose(param_val, old_params_local[param_name], rtol=0, atol=1e-5)
    torch.seed()  # resetting the seed at the end, just to be safe
