import pytest
import torch

from fl4health.clients.apfl_client import ApflClient
from fl4health.model_bases.apfl_base import APFLModule
from tests.clients.fixtures import get_client  # noqa
from tests.test_utils.models_for_test import SmallCnn


@pytest.mark.parametrize("type,model", [(ApflClient, APFLModule(SmallCnn()))])
def test_split_optimizer(get_client: ApflClient) -> None:  # noqa
    apfl_client = get_client

    global_optimizer, local_optimizer = apfl_client.split_optimizer(apfl_client.optimizer)

    # Check that global_optimizer and local_optimizer dont reference the same object
    assert global_optimizer is not local_optimizer

    # Check that the param_groups are equivalent since the local and global models are exact copies
    # at the start
    global_param_groups = global_optimizer.param_groups
    local_param_groups = local_optimizer.param_groups
    for global_group, local_group in zip(global_param_groups, local_param_groups):
        for (global_key, global_vals), (local_key, local_vals) in zip(global_group.items(), local_group.items()):
            assert local_key == global_key
            assert type(local_vals) == type(global_vals)
            # Either Parameter Group or float representing lr
            if isinstance(global_vals, list):
                for global_val, local_val in zip(global_vals, local_vals):
                    assert torch.equal(global_val, local_val)
            else:
                assert global_vals == local_vals
