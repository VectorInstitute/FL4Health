from pathlib import Path

import pytest
import torch
import torch.nn as nn
from flwr.common.parameter import ndarrays_to_parameters

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.server.base_server import FlServer, FlServerWithCheckpointing
from tests.test_utils.models_for_test import LinearTransform

model = LinearTransform()


class DummyFLServer(FlServer):
    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        return model


def test_seed_setting() -> None:
    fl_server = FlServer(PoissonSamplingClientManager(), None, None, None)
    assert fl_server.seed == 2023


def test_no_hydration_with_checkpointer(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestMetricTorchCheckpointer(str(checkpoint_dir), "best_model.pkl", maximize=False)

    # Checkpointer is defined but there is no server-side model hydration to produce a model from the server state.
    # This is not a deal breaker, but may be unintended behavior and the user should be warned
    fl_server_no_hydration = FlServer(PoissonSamplingClientManager(), None, None, checkpointer)
    fl_server_no_hydration._maybe_checkpoint(1.0, server_round=1)
    assert "Server model hydration is not defined" in caplog.text


def test_no_checkpointer_maybe_checkpoint(caplog: pytest.LogCaptureFixture) -> None:
    fl_server_no_checkpointer = FlServer(PoissonSamplingClientManager(), None, None, None)

    # Neither checkpointing nor hydration is defined, we'll have no server-side checkpointing for the FL run.
    fl_server_no_checkpointer._maybe_checkpoint(1.0, server_round=1)
    assert "No checkpointer present. Models will not be checkpointed on server-side." in caplog.text


def test_hydration_and_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestMetricTorchCheckpointer(str(checkpoint_dir), "best_model.pkl", maximize=False)

    # Server-side hydration to convert server state to model and checkpointing behavior are both defined, a model
    # should be saved and be loaded successfully.
    fl_server_both = DummyFLServer(PoissonSamplingClientManager(), None, None, checkpointer)
    fl_server_both._maybe_checkpoint(1.0, server_round=5)
    loaded_model = checkpointer.load_best_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the saved model
    assert torch.equal(model.linear.weight, loaded_model.linear.weight)


def test_fl_server_with_checkpointing(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestMetricTorchCheckpointer(str(checkpoint_dir), "best_model.pkl", maximize=False)
    # Initial model held by server
    initial_model = LinearTransform()
    # represents the model computed by the clients aggregation
    updated_model = LinearTransform()
    parameter_exchanger = FullParameterExchanger()

    server = FlServerWithCheckpointing(
        PoissonSamplingClientManager(), initial_model, parameter_exchanger, None, None, checkpointer
    )
    # Parameters after aggregation (i.e. the updated server-side model)
    server.parameters = ndarrays_to_parameters(parameter_exchanger.push_parameters(updated_model))

    server._maybe_checkpoint(1.0, server_round=5)
    loaded_model = checkpointer.load_best_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the saved model
    assert torch.equal(updated_model.linear.weight, loaded_model.linear.weight)
