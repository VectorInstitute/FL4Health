import pytest
import torch
from opacus import GradSampleModule, PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl4health.utils.privacy_utilities import map_model_to_opacus_model, privacy_validate_and_fix_modules
from tests.test_utils.models_for_test import MnistNetWithBnAndFrozen


model = MnistNetWithBnAndFrozen(True)


def mock_privacy_engine_wrap(model: nn.Module) -> GradSampleModule:
    dataset = TensorDataset(torch.randn(4, 3, 2, 2), torch.randn(4))
    dataloader = DataLoader(dataset, batch_size=2)

    privacy_engine = PrivacyEngine()
    model, _ = privacy_validate_and_fix_modules(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    opacus_model, _, _ = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dataloader, noise_multiplier=1.0, max_grad_norm=1.0
    )
    return opacus_model


def test_map_model_to_opacus(caplog: pytest.LogCaptureFixture) -> None:
    opacus_model = map_model_to_opacus_model(model)
    assert (
        "Provided model is already of type GradSampleModule, skipping conversion to Opacus model type"
        not in caplog.text
    )
    assert isinstance(opacus_model, GradSampleModule)
    # Make sure that the model no longer has any batch norm layers!
    for _, module in opacus_model.named_modules():
        assert not isinstance(module, nn.BatchNorm2d)
    # Mapping it again should produce a log that it is already an opacus model

    _ = map_model_to_opacus_model(opacus_model)
    target_log = (
        "Provided model is already of type <class 'opacus.grad_sample.grad_sample_module.GradSampleModule'>, "
        "skipping conversion to Opacus model type"
    )
    assert target_log in caplog.text


def test_privacy_engine_wrap_model_are_equivalent() -> None:
    wrapped_model = map_model_to_opacus_model(model)
    make_private_model = mock_privacy_engine_wrap(model)

    make_private_model_named_params = list(make_private_model.named_parameters())
    wrapped_model_named_params = list(wrapped_model.named_parameters())
    assert len(make_private_model_named_params) > 0
    assert len(make_private_model_named_params) == len(wrapped_model_named_params)
    for (make_private_name, make_private_params), (wrapped_name, wrapped_params) in zip(
        make_private_model_named_params, wrapped_model_named_params
    ):
        assert make_private_name == wrapped_name
        assert torch.allclose(make_private_params, wrapped_params, atol=0.00001)
