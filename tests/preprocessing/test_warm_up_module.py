import copy

from fl4health.model_bases.apfl_base import ApflModule
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.moon_base import MoonModel
from fl4health.preprocessing.warmed_up_module import WarmedUpModule
from tests.test_utils.models_for_test import FeatureCnn, FendaHeadCnn, HeadCnn, SmallCnn, ToyConvNet, ToyConvNet_2


def test_warm_up_module_loading_similar_models_without_mapping() -> None:
    pretrained_model = ToyConvNet()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.load_from_pretrained(model)

    # Check if the weights are the same with pretrained model after loading
    for model_value, pretrained_model_value in zip(
        model.state_dict().values(), pretrained_model.state_dict().values()
    ):
        assert (model_value == pretrained_model_value).all()

    # Check if the weights are different from previous model after loading
    for old_model_value, model_value in zip(old_model.state_dict().values(), model.state_dict().values()):
        assert (old_model_value != model_value).any()


def test_warm_up_module_loading_different_models_without_mapping() -> None:
    pretrained_model = SmallCnn()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.load_from_pretrained(model)

    # Check if the weights are same with previous model as loading should not have any effect
    for old_model_value, model_value in zip(old_model.state_dict().values(), model.state_dict().values()):
        assert (old_model_value == model_value).any()


def test_partial_warm_up_module_loading_different_models_without_mapping() -> None:
    pretrained_model = ToyConvNet_2()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.load_from_pretrained(model)

    # Check if the weights with same size are loaded from pretrained model and if the weights with different size
    # are same with previous model as loading should not have any effect
    for key in model.state_dict().keys():
        if key in ["conv1.weight", "conv1.bias"]:
            assert (model.state_dict()[key] == pretrained_model.state_dict()[key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_partial_warm_up_module_loading_similar_models_with_mapping() -> None:
    pretrained_model = ToyConvNet()
    model = ToyConvNet()
    old_model = copy.deepcopy(model)
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.weights_mapping_dict = {"conv1": "conv1", "conv2": "conv2"}
    warmupmodule.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict().keys():
        if key in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
            assert (model.state_dict()[key] == pretrained_model.state_dict()[key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_global_warm_up_module_loading_fenda_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    old_model = copy.deepcopy(model)
    print(pretrained_model.state_dict().keys())
    print(model.state_dict().keys())
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.weights_mapping_dict = {"global_module.conv1": "conv1", "global_module.conv2": "conv2"}
    warmupmodule.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict().keys():
        if key in [
            "global_module.conv1.weight",
            "global_module.conv1.bias",
            "global_module.conv2.weight",
            "global_module.conv2.bias",
        ]:
            matching_key = warmupmodule.get_matching_component(key)
            assert matching_key is not None
            assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()
        else:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()


def test_global_and_local_warm_up_module_loading_fenda_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = FendaModel(FeatureCnn(), FeatureCnn(), FendaHeadCnn())
    old_model = copy.deepcopy(model)
    print(pretrained_model.state_dict().keys())
    print(model.state_dict().keys())
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.weights_mapping_dict = {
        "global_module.conv1": "conv1",
        "global_module.conv2": "conv2",
        "local_module.conv1": "conv1",
        "local_module.conv2": "conv2",
    }
    warmupmodule.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model and if the weights not in mapping
    # are same with previous model as loading should not have any effect
    for key in model.state_dict().keys():
        if key in ["model_head.fc1.weight", "model_head.fc1.bias"]:
            assert (model.state_dict()[key] == old_model.state_dict()[key]).all()
        else:
            matching_key = warmupmodule.get_matching_component(key)
            assert matching_key is not None
            assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()


def test_warm_up_module_loading_apfl_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = ApflModule(SmallCnn(), False)
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.weights_mapping_dict = {
        "local_model.conv1": "conv1",
        "local_model.conv2": "conv2",
        "local_model.fc1": "fc1",
        "global_model.conv1": "conv1",
        "global_model.conv2": "conv2",
        "global_model.fc1": "fc1",
    }
    warmupmodule.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model
    for key in model.state_dict().keys():
        matching_key = warmupmodule.get_matching_component(key)
        assert matching_key is not None
        assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()


def test_warm_up_module_loading_moon_model_with_mapping() -> None:
    pretrained_model = SmallCnn()
    model = MoonModel(FeatureCnn(), HeadCnn())
    warmupmodule = WarmedUpModule(pretrained_model=pretrained_model)
    warmupmodule.weights_mapping_dict = {
        "base_module.conv1": "conv1",
        "base_module.conv2": "conv2",
        "head_module.fc1": "fc1",
    }
    warmupmodule.load_from_pretrained(model)

    # Check if only the weights in mapping are loaded from pretrained model
    for key in model.state_dict().keys():
        matching_key = warmupmodule.get_matching_component(key)
        assert matching_key is not None
        assert (model.state_dict()[key] == pretrained_model.state_dict()[matching_key]).all()
