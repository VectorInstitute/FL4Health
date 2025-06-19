import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from flwr.common import Code, EvaluateRes, Status
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg
from freezegun import freeze_time
from peft import LoraConfig, get_peft_model

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.checkpointing.state_checkpointer import ServerStateCheckpointer
from fl4health.client_managers.base_sampling_manager import SimpleClientManager
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.metrics.base_metrics import TEST_LOSS_KEY, TEST_NUM_EXAMPLES_KEY, MetricPrefix
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.reporting import JsonReporter
from fl4health.servers.base_server import FlServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.peft_parameter_extraction import get_all_peft_parameters_from_model
from tests.test_utils.assert_metrics_dict import assert_metrics_dict
from tests.test_utils.custom_client_proxy import CustomClientProxy
from tests.test_utils.models_for_test import LinearTransform


model = LinearTransform()


def test_hydration_no_model_with_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    state_checkpointer = ServerStateCheckpointer(checkpoint_dir=checkpoint_dir)
    # Checkpointer is defined but there is no server-side model defined to produce a model from the server state.
    # An assertion error should be throw stating this
    with pytest.raises(AssertionError) as assertion_error:
        BaseServerCheckpointAndStateModule(
            model=None,
            parameter_exchanger=None,
            model_checkpointers=checkpointer,
            state_checkpointer=state_checkpointer,
        )
    assert "Checkpointer(s) is (are) defined but no model is defined to hydrate" in str(assertion_error.value)


def test_hydration_no_exchanger_with_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    # Checkpointer is defined but there is no parameter exchanger defined to produce a model from the server state.
    # An assertion error should be throw stating this
    with pytest.raises(AssertionError) as assertion_error:
        BaseServerCheckpointAndStateModule(model=model, parameter_exchanger=None, model_checkpointers=checkpointer)
    assert "Checkpointer(s) is (are) defined but no parameter_exchanger is defined to hydrate." in str(
        assertion_error.value
    )


def test_no_checkpointer_maybe_checkpoint(caplog: pytest.LogCaptureFixture) -> None:
    fl_server_no_checkpointer = FlServer(
        client_manager=PoissonSamplingClientManager(), fl_config={}, checkpoint_and_state_module=None
    )

    # Neither checkpointing nor hydration is defined, we'll have no server-side checkpointing for the FL run.
    fl_server_no_checkpointer._maybe_checkpoint(1.0, {}, server_round=1)
    assert "No model checkpointers specified. Skipping any checkpointing." in caplog.text


def test_hydration_and_checkpointer(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model, parameter_exchanger=FullParameterExchanger(), model_checkpointers=checkpointer
    )

    # Server-side hydration to convert server state to model and checkpointing behavior are both defined, a model
    # should be saved and be loaded successfully.
    fl_server_both = FlServer(
        client_manager=PoissonSamplingClientManager(),
        fl_config={},
        checkpoint_and_state_module=checkpoint_and_state_module,
    )
    # Need to mock set the parameters as no FL or exchange is happening.
    fl_server_both.parameters = get_all_model_parameters(model)

    fl_server_both._maybe_checkpoint(1.0, {}, server_round=5)
    loaded_model = checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the saved model
    assert torch.equal(model.linear.weight, loaded_model.linear.weight)


def test_get_peft_parameters() -> None:
    # Add peft parameters to the model
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["linear"],
    )
    peft_model = get_peft_model(model, peft_config)  # type: ignore

    # Extracting peft model parameters and converting to NDArrays object
    server_model = parameters_to_ndarrays(get_all_peft_parameters_from_model(peft_model))

    # Extracted parameters should be a non-empty list of NDArray
    assert isinstance(server_model, list)
    assert len(server_model) > 0


def test_fl_server_with_checkpointing(tmp_path: Path) -> None:
    # Temporary path to write pkl to, will be cleaned up at the end of the test.
    checkpoint_dir = tmp_path.joinpath("resources")
    checkpoint_dir.mkdir()
    checkpointer = BestLossTorchModuleCheckpointer(str(checkpoint_dir), "best_model.pkl")
    # Initial model held by server
    initial_model = LinearTransform()
    # represents the model computed by the clients aggregation
    updated_model = LinearTransform()
    parameter_exchanger = FullParameterExchanger()
    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=initial_model, parameter_exchanger=parameter_exchanger, model_checkpointers=checkpointer
    )

    server = FlServer(
        client_manager=PoissonSamplingClientManager(),
        fl_config={},
        strategy=None,
        checkpoint_and_state_module=checkpoint_and_state_module,
    )
    # Parameters after aggregation (i.e. the updated server-side model)
    server.parameters = ndarrays_to_parameters(parameter_exchanger.push_parameters(updated_model))

    server._maybe_checkpoint(1.0, {}, server_round=5)
    loaded_model = checkpointer.load_checkpoint()
    assert isinstance(loaded_model, LinearTransform)
    # Correct loading tensors of the saved model
    assert torch.equal(updated_model.linear.weight, loaded_model.linear.weight)


@patch("fl4health.servers.base_server.Server.fit")
@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit(mock_fit: Mock) -> None:
    test_history = History()
    test_history.metrics_centralized = {"test_metric1": [(1, 123.123), (2, 123)]}
    test_history.losses_centralized = [(1, 123.123), (2, 123)]
    mock_fit.return_value = (test_history, 1)
    reporter = JsonReporter()
    fl_server = FlServer(client_manager=SimpleClientManager(), fl_config={}, reporters=[reporter])
    fl_server.fit(2, None)
    metrics_to_assert = {
        "host_type": "server",
        "fit_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "fit_end": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
        "rounds": {
            1: {
                "eval_round_metrics_centralized": {"test_metric1": 123.123},
                "val - loss - centralized": 123.123,
            },
            2: {
                "eval_round_metrics_centralized": {"test_metric1": 123},
                "val - loss - centralized": 123,
            },
        },
    }
    errors = assert_metrics_dict(metrics_to_assert, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}, {reporter.metrics}"


@patch("fl4health.servers.base_server.Server.fit_round")
@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_fit_round(mock_fit_round: Mock) -> None:
    test_round = 2
    test_metrics_aggregated = "test metrics aggregated"
    mock_fit_round.return_value = (None, test_metrics_aggregated, None)

    reporter = JsonReporter()
    fl_server = FlServer(client_manager=SimpleClientManager(), fl_config={}, reporters=[reporter])
    fl_server.fit_round(test_round, None)

    metrics_to_assert = {
        "rounds": {
            test_round: {
                "fit_round_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
                "fit_round_metrics": test_metrics_aggregated,
                "fit_round_end": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
            },
        },
    }
    errors = assert_metrics_dict(metrics_to_assert, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}. {reporter.metrics}"


def test_unpack_metrics() -> None:
    # Initialize the server with BasicFedAvg strategy
    strategy = BasicFedAvg(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    fl_server = FlServer(client_manager=Mock(), fl_config={}, strategy=strategy)

    client_proxy = CustomClientProxy("1")
    eval_res = EvaluateRes(
        status=Status(Code.OK, "Success"),
        loss=1.0,
        num_examples=10,
        metrics={
            "val - prediction - accuracy": 0.9,
            TEST_LOSS_KEY: 0.8,
            TEST_NUM_EXAMPLES_KEY: 5,
            f"{MetricPrefix.TEST_PREFIX.value} accuracy": 0.85,
        },
    )

    results: list[tuple[ClientProxy, EvaluateRes]] = [(client_proxy, eval_res)]

    val_results, test_results = fl_server._unpack_metrics(results)

    # Check the validation results
    assert len(val_results) == 1
    assert val_results[0][1].metrics["val - prediction - accuracy"] == 0.9
    assert TEST_LOSS_KEY not in val_results[0][1].metrics

    # Check the test results
    assert len(test_results) == 1
    assert test_results[0][1].metrics[f"{MetricPrefix.TEST_PREFIX.value} accuracy"] == 0.85
    assert test_results[0][1].loss == 0.8


def test_handle_result_aggregation() -> None:
    # Initialize the server with BasicFedAvg strategy
    strategy = BasicFedAvg(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn)
    fl_server = FlServer(client_manager=Mock(), fl_config={}, strategy=strategy)

    client_proxy1 = CustomClientProxy("1")
    eval_res1 = EvaluateRes(
        status=Status(Code.OK, "Success"),
        loss=1.0,
        num_examples=10,
        metrics={
            "val - prediction - accuracy": 0.9,
            TEST_LOSS_KEY: 0.8,
            TEST_NUM_EXAMPLES_KEY: 5,
            f"{MetricPrefix.TEST_PREFIX.value} accuracy": 0.85,
        },
    )
    client_proxy2 = CustomClientProxy("2")
    eval_res2 = EvaluateRes(
        status=Status(Code.OK, "Success"),
        loss=2.0,
        num_examples=20,
        metrics={
            "val - prediction - accuracy": 0.8,
            TEST_LOSS_KEY: 1.6,
            TEST_NUM_EXAMPLES_KEY: 10,
            f"{MetricPrefix.TEST_PREFIX.value} accuracy": 0.75,
        },
    )

    results: list[tuple[ClientProxy, EvaluateRes]] = [
        (client_proxy1, eval_res1),
        (client_proxy2, eval_res2),
    ]
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []

    server_round = 1
    _, val_metrics_aggregated = fl_server._handle_result_aggregation(server_round, results, failures)

    # Check the aggregated validation metrics
    assert "val - prediction - accuracy" in val_metrics_aggregated
    assert val_metrics_aggregated["val - prediction - accuracy"] == pytest.approx(0.8333, rel=1e-3)

    # Check the aggregated test metrics
    assert f"{MetricPrefix.TEST_PREFIX.value} accuracy" in val_metrics_aggregated
    assert val_metrics_aggregated[f"{MetricPrefix.TEST_PREFIX.value} accuracy"] == pytest.approx(0.7833, rel=1e-3)
    assert f"{MetricPrefix.TEST_PREFIX.value} loss - aggregated" in val_metrics_aggregated
    assert val_metrics_aggregated[f"{MetricPrefix.TEST_PREFIX.value} loss - aggregated"] == pytest.approx(
        1.333, rel=1e-3
    )


@patch("fl4health.servers.base_server.FlServer._evaluate_round")
@freeze_time("2012-12-12 12:12:12")
def test_metrics_reporter_evaluate_round(mock_evaluate_round: Mock) -> None:
    test_round = 2
    test_loss_aggregated = "test loss aggregated"
    test_metrics_aggregated = "test metrics aggregated"
    mock_evaluate_round.return_value = (
        test_loss_aggregated,
        test_metrics_aggregated,
        (None, None),
    )
    client_manager = SimpleClientManager()
    client_manager.register(CustomClientProxy("test_id", 1))
    reporter = JsonReporter()
    fl_server = FlServer(
        client_manager=client_manager,
        fl_config={},
        reporters=[reporter],
        strategy=FedAvg(min_evaluate_clients=1, min_available_clients=1),
    )
    fl_server.evaluate_round(test_round, None)

    metrics_to_assert = {
        "rounds": {
            test_round: {
                "eval_round_start": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
                "val - loss - aggregated": test_loss_aggregated,
                "eval_round_metrics_aggregated": test_metrics_aggregated,
                "eval_round_end": str(datetime.datetime(2012, 12, 12, 12, 12, 12)),
            },
        },
    }
    errors = assert_metrics_dict(metrics_to_assert, reporter.metrics)
    assert len(errors) == 0, f"Metrics check failed. Errors: {errors}"
