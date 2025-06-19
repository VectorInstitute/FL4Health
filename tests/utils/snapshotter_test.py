import copy

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fl4health.utils.config import narrow_dict_type
from fl4health.utils.snapshotter import (
    LRSchedulerSnapshotter,
    OptimizerSnapshotter,
    TorchModuleSnapshotter,
)
from tests.test_utils.models_for_test import SingleLayerWithSeed


def compare_mixed_dictionaries(
    dict1: dict[str, torch.Tensor | float | int | list | dict],
    dict2: dict[str, torch.Tensor | float | int | list | dict],
) -> bool:
    if dict1.keys() != dict2.keys():
        return False

    for key, dict1_value in dict1.items():
        if isinstance(dict1_value, torch.Tensor):
            if not torch.equal(dict1_value, narrow_dict_type(dict2, key, torch.Tensor)):
                return False
        elif isinstance(dict1_value, (float, int)):
            dict2_value = dict2[key]
            assert isinstance(dict2_value, (float, int))
            if dict1_value != dict2_value:
                return False
        elif isinstance(dict1_value, list):
            if dict1_value != narrow_dict_type(dict2, key, list):
                return False
        elif isinstance(dict1_value, dict):
            if not compare_mixed_dictionaries(dict1_value, narrow_dict_type(dict2, key, dict)):
                return False
        else:
            raise TypeError(f"Unsupported type in dictionary: {type(dict1_value)}")

    return True


def test_optimizer_lr_model_snapshotters() -> None:
    # Define several optimizers for a client
    local_model = SingleLayerWithSeed()
    global_model = SingleLayerWithSeed(seed=36)
    optimizers: dict[str, Optimizer] = {
        "local": torch.optim.Adam(local_model.parameters(), lr=0.001),
        "global": torch.optim.Adam(global_model.parameters(), lr=0.01),
    }
    lr_schedulers: dict[str, LRScheduler] = {
        "local": torch.optim.lr_scheduler.StepLR(optimizers["local"], step_size=30, gamma=0.1),
        "global": torch.optim.lr_scheduler.StepLR(optimizers["global"], step_size=30, gamma=0.1),
    }
    models: dict[str, nn.Module] = {
        "local": local_model,
        "global": global_model,
    }
    input_data = torch.randn(32, 100)
    target_data = torch.randn(32, 2)
    local_output = local_model(input_data)
    global_output = global_model(input_data)

    criterion = torch.nn.BCEWithLogitsLoss()
    local_loss = criterion(local_output, target_data)
    global_loss = criterion(global_output, target_data)

    local_loss.backward()
    global_loss.backward()

    optimizers["global"].step()
    optimizers["local"].step()

    lr_schedulers["local"].step()
    lr_schedulers["global"].step()
    # Keep a copy of the client state as reference
    old_optimizers = copy.deepcopy(optimizers)
    old_lr_schedulers = copy.deepcopy(lr_schedulers)
    old_models = copy.deepcopy(models)

    # snapshot client attribute that we want to save
    optimizer_snapshotter = OptimizerSnapshotter()
    optimizer_dict_to_be_saved = optimizer_snapshotter.save_attribute(attribute=optimizers)
    model_snapshotter = TorchModuleSnapshotter()
    model_dict_to_be_saved = model_snapshotter.save_attribute(attribute=models)
    lr_scheduler_snapshotter = LRSchedulerSnapshotter()
    lr_scheduler_dict_to_be_saved = lr_scheduler_snapshotter.save_attribute(attribute=lr_schedulers)

    # Now create new models, optimizers, and rl_schedulers
    local_model_new = SingleLayerWithSeed(seed=4)
    global_model_new = SingleLayerWithSeed(seed=5)
    # New optimizers
    new_optimizers: dict[str, Optimizer] = {
        "local": torch.optim.Adam(local_model_new.parameters(), lr=0.001),
        "global": torch.optim.Adam(global_model_new.parameters(), lr=0.01),
    }
    # New lr_schedulers
    new_lr_schedulers: dict[str, LRScheduler] = {
        "local": torch.optim.lr_scheduler.StepLR(optimizers["local"], step_size=30, gamma=0.1),
        "global": torch.optim.lr_scheduler.StepLR(optimizers["global"], step_size=30, gamma=0.1),
    }
    new_models: dict[str, nn.Module] = {
        "local": local_model_new,
        "global": global_model_new,
    }

    for key, value in new_models.items():
        assert not compare_mixed_dictionaries(value.state_dict(), old_models[key].state_dict())

    for key, optimizer in new_optimizers.items():
        assert not compare_mixed_dictionaries(
            optimizer.state_dict()["state"], old_optimizers[key].state_dict()["state"]
        )

    for key, schedulers in new_lr_schedulers.items():
        assert not compare_mixed_dictionaries(schedulers.state_dict(), old_lr_schedulers[key].state_dict())

    # Load the state
    optimizer_snapshotter.load_attribute(optimizer_dict_to_be_saved, new_optimizers)
    model_snapshotter.load_attribute(model_dict_to_be_saved, new_models)
    lr_scheduler_snapshotter.load_attribute(lr_scheduler_dict_to_be_saved, new_lr_schedulers)

    # Check that the state of the new optimizers are the same as the ones saved.
    for optimizer_type, new_optimizer in new_optimizers.items():
        assert compare_mixed_dictionaries(
            new_optimizer.state_dict()["state"], old_optimizers[optimizer_type].state_dict()["state"]
        )
    # Check that the state of the new models are the same as the ones saved.
    for model_type, new_model in new_models.items():
        assert compare_mixed_dictionaries(new_model.state_dict(), old_models[model_type].state_dict())

    # Check that the state of the new lr_schedulers are the same as the ones saved.
    for scheduler_type, new_scheduler in new_lr_schedulers.items():
        assert compare_mixed_dictionaries(new_scheduler.state_dict(), old_lr_schedulers[scheduler_type].state_dict())
