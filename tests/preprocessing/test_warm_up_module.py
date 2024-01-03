import copy

from fl4health.preprocessing.warmed_up_module import WarmedUpModule
from tests.test_utils.models_for_test import SmallCnn, ToyConvNet


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
