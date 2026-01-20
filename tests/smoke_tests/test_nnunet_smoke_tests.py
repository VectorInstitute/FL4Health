import asyncio

import pytest

from tests.smoke_tests.run_smoke_test import (
    run_smoke_test,
)


# Marks all test coroutines in this module
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def try_running_test_task(task: asyncio.Task) -> None:
    """
    Helper for running task.

    If an exception is reached, then cancel the task and wait for it to be cleared.
    """
    try:
        await task
    except Exception as e:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)  # allow time to clean up cancelled task
        pytest.fail(f"Smoke test failed due to error. {e}")


def assert_on_done_task(task: asyncio.Task) -> None:
    """
    This function takes a done task and makes assert if a result was returned.

    If an exception was returned, then it fails the pytest for proper shutdown.
    Also, if the task was cancelled, then it cleans up the cancelled tasks so the
    next test doesn't get this hangover and fails as a result.
    """
    e = task.exception()  # handle TimeoutError / CancelledError above this func
    # at this point there is either an Exception or a Result and the task wasn't cancelled
    if e:
        pytest.fail(f"Smoke test execution failed: {e}")
    else:
        server_errors, client_errors = task.result()
        assert len(server_errors) == 0, f"Server metrics check failed. Errors: {server_errors}"
        assert len(client_errors) == 0, f"Client metrics check failed. Errors: {client_errors}"


@pytest.mark.smoketest
async def test_nnunet_config_2d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client",
        config_path="tests/smoke_tests/nnunet_config_2d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_nnunet_config_3d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client",
        config_path="tests/smoke_tests/nnunet_config_3d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_flexible_nnunet_config_2d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client_flexible",
        config_path="tests/smoke_tests/nnunet_config_2d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_flexible_nnunet_config_3d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_example.server",
        client_python_path="examples.nnunet_example.client_flexible",
        config_path="tests/smoke_tests/nnunet_config_3d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_ditto_flexible_nnunet_config_2d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_pfl_example.server",
        client_python_path="examples.nnunet_pfl_example.client",
        config_path="tests/smoke_tests/nnunet_config_2d.yaml",
        dataset_path="examples/datasets/nnunet",
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_nnunet_pfl_mr_mtl_config_3d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_pfl_example.server",
        client_python_path="examples.nnunet_pfl_example.client",
        config_path="tests/smoke_tests/nnunet_config_3d.yaml",
        dataset_path="examples/datasets/nnunet",
        additional_client_args={"--personalized_strategy": "mr_mtl"},
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)


@pytest.mark.smoketest
async def test_nnunet_pfl_mr_mtl_config_2d(tolerance: float) -> None:
    coroutine = run_smoke_test(  # By default will use Task04_Hippocampus Dataset
        server_python_path="examples.nnunet_pfl_example.server",
        client_python_path="examples.nnunet_pfl_example.client",
        config_path="tests/smoke_tests/nnunet_config_2d.yaml",
        dataset_path="examples/datasets/nnunet",
        additional_client_args={"--personalized_strategy": "mr_mtl"},
        tolerance=tolerance,
        read_logs_timeout=450,
    )
    task = asyncio.create_task(coroutine)
    await try_running_test_task(task)
    assert_on_done_task(task)
