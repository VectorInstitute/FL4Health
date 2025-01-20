import os
from pathlib import Path

import pytest

from .run_smoke_test import load_metrics_from_file, run_fault_tolerance_smoke_test, run_smoke_test

# skip some tests that currently fail if running locallly
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_basic_server_client_cifar(tolerance: float, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    await run_fault_tolerance_smoke_test(
        server_python_path="tests.smoke_tests.load_from_checkpoint_example.server",
        client_python_path="tests.smoke_tests.load_from_checkpoint_example.client",
        config_path="tests/smoke_tests/load_from_checkpoint_example/config.yaml",
        partial_config_path="tests/smoke_tests/load_from_checkpoint_example/partial_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        seed=42,
        server_metrics=load_metrics_from_file("tests/smoke_tests/basic_server_metrics.json"),
        client_metrics=load_metrics_from_file("tests/smoke_tests/basic_client_metrics.json"),
        intermediate_checkpoint_dir=checkpoint_dir.as_posix(),
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_nnunet_config_2d(tolerance: float) -> None:
    await run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client",
        config_path="tests/smoke_tests/nnunet_config_2d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_nnunet_config_3d(tolerance: float) -> None:
    await run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client",
        config_path="tests/smoke_tests/nnunet_config_3d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_scaffold(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.scaffold_example.server",
        client_python_path="examples.scaffold_example.client",
        config_path="tests/smoke_tests/scaffold_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        seed=42,
        server_metrics=load_metrics_from_file("tests/smoke_tests/scaffold_server_metrics.json"),
        client_metrics=load_metrics_from_file("tests/smoke_tests/scaffold_client_metrics.json"),
        tolerance=tolerance,
    )


@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Test doesn't work locally.")
@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_apfl(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.apfl_example.server",
        client_python_path="examples.apfl_example.client",
        config_path="tests/smoke_tests/apfl_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        seed=42,
        server_metrics=load_metrics_from_file("tests/smoke_tests/apfl_server_metrics.json"),
        client_metrics=load_metrics_from_file("tests/smoke_tests/apfl_client_metrics.json"),
        tolerance=tolerance,
    )


@pytest.mark.skipif(not IN_GITHUB_ACTIONS, reason="Test doesn't work locally.")
@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_feddg_ga(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.feddg_ga_example.server",
        client_python_path="examples.feddg_ga_example.client",
        config_path="tests/smoke_tests/feddg_ga_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        seed=42,
        server_metrics=load_metrics_from_file("tests/smoke_tests/feddg_ga_server_metrics.json"),
        client_metrics=load_metrics_from_file("tests/smoke_tests/feddg_ga_client_metrics.json"),
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_basic(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.basic_example.server",
        client_python_path="examples.basic_example.client",
        config_path="tests/smoke_tests/basic_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_client_level_dp_cifar(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.dp_fed_examples.client_level_dp.server",
        client_python_path="examples.dp_fed_examples.client_level_dp.client",
        config_path="tests/smoke_tests/client_level_dp_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        skip_assert_client_fl_rounds=True,
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_client_level_dp_breast_cancer(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.dp_fed_examples.client_level_dp_weighted.server",
        client_python_path="examples.dp_fed_examples.client_level_dp_weighted.client",
        config_path="tests/smoke_tests/client_level_dp_weighted_config.yaml",
        dataset_path="examples/datasets/breast_cancer_data/hospital_0.csv",
        skip_assert_client_fl_rounds=True,
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_instance_level_dp_cifar(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.dp_fed_examples.instance_level_dp.server",
        client_python_path="examples.dp_fed_examples.instance_level_dp.client",
        config_path="tests/smoke_tests/instance_level_dp_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        skip_assert_client_fl_rounds=True,
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_dp_scaffold(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.dp_scaffold_example.server",
        client_python_path="examples.dp_scaffold_example.client",
        config_path="tests/smoke_tests/dp_scaffold_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fedbn(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fedbn_example.server",
        client_python_path="examples.fedbn_example.client",
        config_path="tests/smoke_tests/fedbn_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fed_eval(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.federated_eval_example.server",
        client_python_path="examples.federated_eval_example.client",
        config_path="tests/smoke_tests/federated_eval_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        checkpoint_path="examples/assets/fed_eval_example/best_checkpoint_fczjmljm.pkl",
        assert_evaluation_logs=True,
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fedper_mnist(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fedper_example.server",
        client_python_path="examples.fedper_example.client",
        config_path="tests/smoke_tests/fedper_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fedper_cifar(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fedrep_example.server",
        client_python_path="examples.fedrep_example.client",
        config_path="tests/smoke_tests/fedrep_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_ditto_mnist() -> None:
    await run_smoke_test(
        server_python_path="examples.ditto_example.server",
        client_python_path="examples.ditto_example.client",
        config_path="tests/smoke_tests/ditto_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_mr_mtl_mnist(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.mr_mtl_example.server",
        client_python_path="examples.mr_mtl_example.client",
        config_path="tests/smoke_tests/mr_mtl_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fenda(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fenda_example.server",
        client_python_path="examples.fenda_example.client",
        config_path="tests/smoke_tests/fenda_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fenda_ditto(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fenda_ditto_example.server",
        client_python_path="examples.fenda_ditto_example.client",
        config_path="tests/smoke_tests/fenda_ditto_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        checkpoint_path="examples/assets/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_perfcl(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.perfcl_example.server",
        client_python_path="examples.perfcl_example.client",
        config_path="tests/smoke_tests/perfcl_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_fl_plus_local(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.fl_plus_local_ft_example.server",
        client_python_path="examples.fl_plus_local_ft_example.client",
        config_path="tests/smoke_tests/fl_plus_local_ft_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_moon(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.moon_example.server",
        client_python_path="examples.moon_example.client",
        config_path="tests/smoke_tests/moon_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_ensemble(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.ensemble_example.server",
        client_python_path="examples.ensemble_example.client",
        config_path="tests/smoke_tests/ensemble_config.yaml",
        dataset_path="examples/datasets/mnist_data/",
        tolerance=tolerance,
    )


@pytest.mark.smoketest
@pytest.mark.asyncio()
async def test_flash(tolerance: float) -> None:
    await run_smoke_test(
        server_python_path="examples.flash_example.server",
        client_python_path="examples.flash_example.client",
        config_path="tests/smoke_tests/flash_config.yaml",
        dataset_path="examples/datasets/cifar_data/",
        tolerance=tolerance,
    )
