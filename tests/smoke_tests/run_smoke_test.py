import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from flwr.common.typing import Config
from six.moves import urllib

from examples.fedprox_example.client import MnistFedProxClient
from fl4health.utils.metrics import Accuracy

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


async def run_smoke_test(
    server_python_path: str,
    client_python_path: str,
    config_path: str,
    dataset_path: str,
    checkpoint_path: Optional[str] = None,
    assert_evaluation_logs: Optional[bool] = False,
    # The param below exists to work around an issue with some clients
    # not printing the "Current FL Round" log message reliably
    skip_assert_client_fl_rounds: Optional[bool] = False,
) -> None:
    logger.info("Running smoke tests with parameters:")
    logger.info(f"\tServer : {server_python_path}")
    logger.info(f"\tClient : {client_python_path}")
    logger.info(f"\tConfig : {config_path}")
    logger.info(f"\tDataset: {dataset_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    _preload_dataset(dataset_path, config)

    # Start the server and capture its process object
    logger.info("Starting server...")
    server_process = await asyncio.create_subprocess_exec(
        "python",
        "-m",
        server_python_path,
        "--config_path",
        config_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # reads lines from the server output in search of the startup log message
    # times out after 20s of inactivity if it doesn't find the log message
    full_server_output = ""
    startup_messages = [
        "FL starting",
        "Using Warm Start Strategy. Waiting for clients to be available for polling",
        "Polling Clients for sample counts",
        "Federated Evaluation Starting",
    ]

    output_found = False
    while not output_found:
        try:
            assert server_process.stdout is not None, "Server's process stdout is None"
            server_output_in_bytes = await asyncio.wait_for(server_process.stdout.readline(), 20)
            server_output = server_output_in_bytes.decode()
            logger.debug(f"Server output: {server_output}")
            full_server_output += server_output
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for server startup messages")
            break

        return_code = server_process.returncode
        assert return_code is None or (return_code is not None and return_code == 0), (
            f"Full output:\n{full_server_output}\n" f"[ASSERT ERROR] Server exited with code {return_code}."
        )

        if any(startup_message in server_output for startup_message in startup_messages):
            output_found = True

    assert output_found, (
        f"Full output:\n{full_server_output}\n" f"[ASSERT_ERROR] Startup log message not found in server output."
    )

    logger.info("Server started")

    # Start n number of clients and capture their process objects
    client_processes = []
    for i in range(config["n_clients"]):
        logger.info(f"Starting client {i}")

        client_args = ["-m", client_python_path, "--dataset_path", dataset_path]
        if checkpoint_path is not None:
            client_args.append("--checkpoint_path")
            client_args.append(checkpoint_path)

        client_process = await asyncio.create_subprocess_exec(
            "python",
            *client_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        client_processes.append(client_process)

    # Collecting the clients output when their processes finish
    full_client_outputs = []
    for i in range(len(client_processes)):
        full_client_outputs.append(
            await _wait_for_process_to_finish_and_retrieve_logs(client_processes[i], f"Client {i}")
        )

    logger.info("All clients finished execution")

    # Collecting the server output when its process finish
    full_server_output = await _wait_for_process_to_finish_and_retrieve_logs(server_process, "Server")

    logger.info("Server has finished execution")

    # server assertions
    assert "error" not in full_server_output.lower(), (
        f"Full output:\n{full_server_output}\n" "[ASSERT ERROR] Error message found for server."
    )
    if assert_evaluation_logs:
        assert f"Federated Evaluation received {config['n_clients']} results and 0 failures" in full_server_output, (
            f"Full output:\n{full_server_output}\n" "[ASSERT ERROR] Last FL round message not found for server."
        )
        assert "Federated Evaluation Finished" in full_server_output, (
            f"Full output:\n{full_server_output}\n"
            "[ASSERT ERROR] Federated Evaluation Finished message not found for server."
        )
    else:
        assert f"evaluate_round {config['n_server_rounds']}" in full_server_output, (
            f"Full output:\n{full_server_output}\n" "[ASSERT ERROR] Last FL round message not found for server."
        )
        assert "FL finished" in full_server_output, (
            f"Full output:\n{full_server_output}\n" "[ASSERT ERROR] FL finished message not found for server."
        )
    assert all(
        message in full_server_output
        for message in [
            "app_fit: losses_distributed",
            "app_fit: metrics_distributed_fit",
            "app_fit: metrics_distributed",
            "app_fit: losses_centralized",
            "app_fit: metrics_centralized",
        ]
    ), f"Full output:\n{full_server_output}\n[ASSERT ERROR] Metrics message not found for server."

    # client assertions
    for i in range(len(full_client_outputs)):
        assert "error" not in full_client_outputs[i].lower(), (
            f"Full client output:\n{full_client_outputs[i]}\n" f"[ASSERT ERROR] Error message found for client {i}."
        )
        assert "Disconnect and shut down" in full_client_outputs[i], (
            f"Full client output:\n{full_client_outputs[i]}\n"
            f"[ASSERT ERROR] Shutdown message not found for client {i}."
        )
        if assert_evaluation_logs:
            assert "Client Evaluation Local Model Metrics" in full_client_outputs[i], (
                f"Full client output:\n{full_client_outputs[i]}\n"
                f"[ASSERT ERROR] 'Client Evaluation Local Model Metrics' message not found for client {i}."
            )
        elif not skip_assert_client_fl_rounds:
            assert f"Current FL Round: {config['n_server_rounds']}" in full_client_outputs[i], (
                f"Full client output:\n{full_client_outputs[i]}\n"
                f"[ASSERT ERROR] Last FL round message not found for client {i}."
            )

    logger.info("All checks passed. Test finished.")


def _preload_dataset(dataset_path: str, config: Config) -> None:
    if "mnist" in dataset_path:
        logger.info("Preloading MNIST dataset...")

        # Working around MNIST download issue
        # https://github.com/pytorch/vision/issues/1938
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        # Creating a client and getting the data loaders will trigger
        # the dataset's download
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        client = MnistFedProxClient(Path(dataset_path), [Accuracy()], device)
        client.get_data_loaders(config)

        logger.info("Finished preloading MNIST dataset")
    else:
        logger.info("Preload not supported for specified dataset. Skipping.")


async def _wait_for_process_to_finish_and_retrieve_logs(
    process: asyncio.subprocess.Process,
    process_name: str,
) -> str:
    logger.info(f"Waiting for {process_name} to finish execution to collect its output...")
    # Times out after 5 minutes just so it doesn't hang for a very long timeon some edge cases
    stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=300)
    logger.info(f"Output collected for {process_name}")

    full_output = stdout_bytes.decode().replace("\\n", "\n")
    logger.debug(f"{process_name} stdout: {full_output}")

    if stderr_bytes is not None and len(stderr_bytes) > 0:
        stderr = stderr_bytes.decode().replace("\\n", "\n")
        full_output += stderr
        logger.error(f"{process_name} stderr: {stderr}")

    # checking for clients with failure exit codes
    return_code = process.returncode
    assert return_code is None or (return_code is not None and return_code == 0), (
        f"Full output:\n{full_output}\n" f"[ASSERT ERROR] {process_name} exited with code {return_code}."
    )

    return full_output


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fedprox_example.server",
            client_python_path="examples.fedprox_example.client",
            config_path="tests/smoke_tests/fedprox_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.scaffold_example.server",
            client_python_path="examples.scaffold_example.client",
            config_path="tests/smoke_tests/scaffold_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.apfl_example.server",
            client_python_path="examples.apfl_example.client",
            config_path="tests/smoke_tests/apfl_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.basic_example.server",
            client_python_path="examples.basic_example.client",
            config_path="tests/smoke_tests/basic_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.dp_fed_examples.client_level_dp.server",
            client_python_path="examples.dp_fed_examples.client_level_dp.client",
            config_path="tests/smoke_tests/client_level_dp_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            skip_assert_client_fl_rounds=True,
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.dp_fed_examples.client_level_dp_weighted.server",
            client_python_path="examples.dp_fed_examples.client_level_dp_weighted.client",
            config_path="tests/smoke_tests/client_level_dp_weighted_config.yaml",
            dataset_path="examples/datasets/breast_cancer_data/hospital_0.csv",
            skip_assert_client_fl_rounds=True,
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.dp_fed_examples.instance_level_dp.server",
            client_python_path="examples.dp_fed_examples.instance_level_dp.client",
            config_path="tests/smoke_tests/instance_level_dp_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            skip_assert_client_fl_rounds=True,
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.dp_scaffold_example.server",
            client_python_path="examples.dp_scaffold_example.client",
            config_path="tests/smoke_tests/dp_scaffold_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fedbn_example.server",
            client_python_path="examples.fedbn_example.client",
            config_path="tests/smoke_tests/fedbn_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.federated_eval_example.server",
            client_python_path="examples.federated_eval_example.client",
            config_path="tests/smoke_tests/federated_eval_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            checkpoint_path="examples/assets/best_checkpoint_fczjmljm.pkl",
            assert_evaluation_logs=True,
        )
    )
    loop.close()
