from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from tests.test_utils.models_for_test import FeatureCnn, HeadCnn


def test_sequentially_split_exchange_base_model_gets_correct_layers() -> None:
    model = SequentiallySplitExchangeBaseModel(FeatureCnn(), HeadCnn())
    layers_to_exchange = model.layers_to_exchange()
    assert len(layers_to_exchange) > 0
    filtered_layer_names = [layer_name for layer_name in model.state_dict() if layer_name.startswith("base_module.")]
    for test_layer, expected_layer in zip(layers_to_exchange, filtered_layer_names):
        assert test_layer == expected_layer
