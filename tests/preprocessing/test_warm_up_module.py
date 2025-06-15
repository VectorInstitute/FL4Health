import copy
import json
import os
from pathlib import Path

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.moon_base import MoonModel
from fl4health.preprocessing.warmed_up_module import WarmedUpModule
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn, HeadCnn, SmallCnn, ToyConvNet, ToyConvNet2


def test_initializing_warm_up_module(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()

    # Save a temporary model using checkpointer
    saved_model = SmallCnn()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    checkpointer.maybe_checkpoint(saved_model, 0.7, {})

    # Save a temporary weights mapping dict
    weights_mapping_path = tmp_path.joinpath("data.json")
    saved_weights_mapping_dict = {
        "base_module.conv1": "conv1",
        "base_module.conv2": "conv2",
        "head_module.fc1": "fc1",
    }
    with open(weights_mapping_path, "w") as fp:
        json.dump(saved_weights_mapping_dict, fp)

    # Load the saved model using warmup module
    warmup_module = WarmedUpModule(
        pretrained_model_path=Path(os.path.join(checkpoint_dir, "best_model.pkl")),
        weights_mapping_path=weights_mapping_path,
    )

    # Check if the pretrained model state is loaded correctly
    assert warmup_module.pretrained_model_state is not None
    for saved_model_value, pretrained_model_value in zip(
        saved_model.state_dict().values(), warmup_module.pretrained_model_state.values()
    ):
        assert (saved_model_value == pretrained_model_value).all()

    # Check if the weights mapping dict is loaded correctly
    assert warmup_module.weights_mapping_dict is not None
    for key in warmup_module.weights_mapping_dict:
        assert warmup_module.weights_mapping_dict[key] == saved_weights_mapping_dict[key]


def test_loading_similar_models_without_mapping() -> None:
    pretrained_model = ToyConvNet()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.load_from_pretrained(model)

    # Check if the weights are the same with pretrained model after loading
    for model_value, pretrained_model_value in zip(
        model.state_dict().values(), pretrained_model.state_dict().values()
    ):
        assert (model_value == pretrained_model_value).all()

    # Check if the weights are different from previous model after loading
    for old_model_value, model_value in zip(old_model.state_dict().values(), model.state_dict().values()):
        assert (old_model_value != model_value).any()


def test_loading_different_models_without_mapping() -> None:
    pretrained_model = SmallCnn()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.load_from_pretrained(model)

    # Check if the weights are same with previous model as loading should not have any effect
    for old_model_value, model_value in zip(old_model.state_dict().values(), model.state_dict().values()):
        assert (old_model_value == model_value).all()


def test_partial_loading_different_models_without_mapping() -> None:
    pretrained_model = ToyConvNet2()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.load_from_pretrained(model)

    # Check if the weights with same size are loaded from pretrained model and if the weights with different size
    # are same with previous model as loading should not have any effect
    for key in model.state_dict():
        if key in ["conv1.weight", "conv1.bias"]:
            assert (model.state_dict()[key] == pretrained_model.state_dict()[key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_partial_loading_similar_models_with_mapping() -> None:
    pretrained_model = ToyConvNet()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.weights_mapping_dict = {"conv1": "conv1", "conv2": "conv2"}
    warmup_module.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict():
        if key in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
            assert (model.state_dict()[key] == pretrained_model.state_dict()[key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_global_loading_fenda_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.weights_mapping_dict = {
        "second_feature_extractor.conv1": "conv1",
        "second_feature_extractor.conv2": "conv2",
    }
    warmup_module.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict():
        if key in [
            "second_feature_extractor.conv1.weight",
            "second_feature_extractor.conv1.bias",
            "second_feature_extractor.conv2.weight",
            "second_feature_extractor.conv2.bias",
        ]:
            matching_key = warmup_module.get_matching_component(key)
            assert matching_key is not None
            assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_global_and_local_loading_fenda_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    old_model = copy.deepcopy(model)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.weights_mapping_dict = {
        "second_feature_extractor.conv1": "conv1",
        "second_feature_extractor.conv2": "conv2",
        "first_feature_extractor.conv1": "conv1",
        "first_feature_extractor.conv2": "conv2",
    }
    warmup_module.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict():
        if key in ["model_head.fc1.weight", "model_head.fc1.bias"]:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()
        else:
            matching_key = warmup_module.get_matching_component(key)
            assert matching_key is not None
            assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()


def test_loading_apfl_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = ApflModule(SmallCnn(), False)
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.weights_mapping_dict = {
        "local_model.conv1": "conv1",
        "local_model.conv2": "conv2",
        "local_model.fc1": "fc1",
        "global_model.conv1": "conv1",
        "global_model.conv2": "conv2",
        "global_model.fc1": "fc1",
    }
    warmup_module.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model
    for key in model.state_dict():
        matching_key = warmup_module.get_matching_component(key)
        assert matching_key is not None
        assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()


def test_loading_moon_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = MoonModel(FeatureCnn(), HeadCnn())
    warmup_module = WarmedUpModule(pretrained_model=pretrained_model)
    warmup_module.weights_mapping_dict = {
        "base_module.conv1": "conv1",
        "base_module.conv2": "conv2",
        "head_module.fc1": "fc1",
    }
    warmup_module.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model
    for key in model.state_dict():
        matching_key = warmup_module.get_matching_component(key)
        assert matching_key is not None
        assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()
