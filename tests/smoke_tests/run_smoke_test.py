import asyncio
import datetime
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from flwr.common.typing import Config
from pytest import approx
from six.moves import urllib

from examples.fedprox_example.client import MnistFedProxClient
from fl4health.utils.load_data import load_cifar10_data
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
    seed: Optional[int] = None,
    server_metrics: Optional[Dict[str, Any]] = None,
    client_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Runs a smoke test for a given server, client, and dataset configuration.

    Uses asyncio to kick off one server instance defined by the `server_python_path` module and N client instances
    defined by the `client_python_path` module (N is defined by the `n_clients` attribute in the config). Waits for the
    clients and server to complete execution and then make assertions on their logs to determine they have completed
    execution successfully.

    Typical usage examples:

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            run_smoke_test(
                server_python_path="examples.fedprox_example.server",
                client_python_path="examples.fedprox_example.client",
                config_path="tests/smoke_tests/fedprox_config.yaml",
                dataset_path="examples/datasets/mnist_data/",
            )
        )
        loop.close()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            run_smoke_test(
                server_python_path="examples.dp_fed_examples.instance_level_dp.server",
                client_python_path="examples.dp_fed_examples.instance_level_dp.client",
                config_path="tests/smoke_tests/instance_level_dp_config.yaml",
                dataset_path="examples/datasets/cifar_data/",
                skip_assert_client_fl_rounds=True,
            )
        )
        loop.close()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            run_smoke_test(
                server_python_path="examples.federated_eval_example.server",
                client_python_path="examples.federated_eval_example.client",
                config_path="tests/smoke_tests/federated_eval_config.yaml",
                dataset_path="examples/datasets/cifar_data/",
                checkpoint_path="examples/assets/best_checkpoint_fczjmljm.pkl",
                assert_evaluation_logs=True,
                seed=42,
                server_metrics={
                    "rounds": {
                        "1": {
                            "metrics_aggregated": {"val - prediction - accuracy": 0.4744},
                             # to override default tolerance, pass it in as a dictionary:
                            "loss_aggregated": {"target_value": 2.001, "custom_tolerance": 0.05},
                        },
                    },
                },
                client_metrics={
                    "rounds": {
                        "2": {
                            "fit_metrics": {"train - prediction - accuracy": (0.2031, 0.005)},
                            "loss_dict": {
                                "checkpoint": {"target_value": 2.1473, "custom_tolerance": 0.05},
                                "backward": 2.1736,
                            }
                        },
                    },
                },
            )
        )
        loop.close()

    Args:
        server_python_path (str): the path for the executable server module
        client_python_path (str): the path for the executable client module
        config_path (str): the path for the config yaml file. The following attributes are required
            by this function:
            `n_clients`: the number of clients to be started
            `n_server_rounds`:  the number of rounds to be ran by the server
            `batch_size`: the size of the batch, to be used by the dataset preloader
        dataset_path (str): the path of the dataset. Depending on which dataset is being used, it will ty to preload it
            to avoid problems when running on different runtimes.
        checkpoint_path (Optional[str]): Optional, default None. If set, it will send that path as a checkpoint model
            to the client.
        assert_evaluation_logs (Optional[bool]): Optional, default `False`. Set this to `True` if testing an
            evaluation model, which produces different log outputs.
        skip_assert_client_fl_rounds (Optional[str]): Optional, default `False`. If set to `True`, will skip the
            assertion of the "Current FL Round" message on the clients' logs. This is necessary because some clients
            (namely client_level_dp, client_level_dp_weighted, instance_level_dp) do not reliably print that message.
        seed (Optional[int]): The random seed to be passed in to both the client and the server.
        server_metrics (Optional[Dict[str, Any]]): A dictionary of metrics to be checked against the metrics file
            saved by the server. Should be in the same format as fl4health.reporting.metrics.MetricsReporter.
            Default is None.
        client_metrics (Optional[Dict[str, Any]]): A dictionary of metrics to be checked against the metrics file
            saved by the clients. Should be in the same format as fl4health.reporting.metrics.MetricsReporter.
            Default is None.
    """
    clear_metrics_folder()

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
    server_args = ["-m", server_python_path, "--config_path", config_path]
    if seed is not None:
        server_args.extend(["--seed", str(seed)])

    server_process = await asyncio.create_subprocess_exec(
        "python",
        *server_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # reads lines from the server output in search of the startup log message
    # times out after 20s of inactivity if it doesn't find the log message
    full_server_output = ""
    startup_messages = [
        # printed by fedprox, apfl, basic_example, fedbn, fedper, fenda, fl_plus_local_ft and moon
        "FL starting",
        # printed by scaffold
        "Using Warm Start Strategy. Waiting for clients to be available for polling",
        # printed by client_level_dp, client_level_dp_weighted, instance_level_dp and dp_scaffold
        "Polling Clients for sample counts",
        # printed by federated_eval
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
            client_args.extend(["--checkpoint_path", checkpoint_path])
        if seed is not None:
            client_args.extend(["--seed", str(seed)])

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

    server_errors = _assert_metrics(MetricType.SERVER, server_metrics)
    assert len(server_errors) == 0, f"Server metrics check failed. Errors: {server_errors}"

    # client assertions
    client_errors = []
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

        client_errors.extend(_assert_metrics(MetricType.CLIENT, client_metrics))
        assert len(client_errors) == 0, f"Client metrics check failed. Errors: {client_errors}"

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

    elif "cifar" in dataset_path:
        logger.info("Preloading CIFAR10 dataset...")
        load_cifar10_data(Path(dataset_path), int(config["batch_size"]))
        logger.info("Finished preloading CIFAR10 dataset")

    else:
        logger.info("Preload not supported for specified dataset. Skipping.")


async def _wait_for_process_to_finish_and_retrieve_logs(
    process: asyncio.subprocess.Process,
    process_name: str,
    timeout: int = 300,  # timeout for the whole process to complete
) -> str:
    logger.info(f"Collecting output for {process_name}...")
    full_output = ""
    try:
        assert process.stdout
        start_time = datetime.datetime.now()
        while True:
            # giving a smaller timeout here just in case it hangs for a long time waiting for a single log line
            output_in_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
            output = output_in_bytes.decode().replace("\\n", "\n")
            logger.debug(f"{process_name} output: {output}")
            full_output += output
            return_code = process.returncode

            if output == "" and return_code is not None:
                break

            elapsed_time = datetime.datetime.now() - start_time
            if elapsed_time.seconds > timeout:
                raise Exception(f"Timeout limit of {timeout}s exceeded waiting for {process_name} to finish execution")

    except Exception as ex:
        logger.error(f"{process_name} output:\n{full_output}")
        logger.exception(f"Error collecting {process_name} log messages:")
        raise ex

    logger.info(f"Output collected for {process_name}")

    # checking for clients with failure exit codes
    assert return_code is None or (return_code is not None and return_code == 0), (
        f"Full output:\n{full_output}\n" f"[ASSERT ERROR] {process_name} exited with code {return_code}."
    )

    return full_output


class MetricType(Enum):
    SERVER = "server"
    CLIENT = "client"


DEFAULT_METRICS_FOLDER = Path("metrics")
DEFAULT_TOLERANCE = 0.0005


def _assert_metrics(metric_type: MetricType, metrics_to_assert: Optional[Dict[str, Any]] = None) -> List[str]:
    errors: List[str] = []
    if metrics_to_assert is None:
        return errors

    metrics_found = False
    for file in DEFAULT_METRICS_FOLDER.iterdir():
        if not file.is_file() or not str(file).endswith(".json"):
            continue

        with open(file) as f:
            metrics = json.load(f)

        if metrics["type"] != metric_type.value:
            continue

        metrics_found = True
        errors.extend(_assert_metrics_dict(metrics_to_assert, metrics))

    if not metrics_found:
        errors.append(f"Metrics of type {metric_type.value} not found.")

    return errors


def _assert_metrics_dict(metrics_to_assert: Dict[str, Any], metrics_saved: Dict[str, Any]) -> List[str]:
    errors = []

    def _assert(value: Any, saved_value: Any) -> Optional[str]:
        # helper function to avoid code repetition
        tolerance = DEFAULT_TOLERANCE
        if isinstance(value, dict):
            # if the value is a dictionary, extract the target value and the custom tolerance
            tolerance = value["custom_tolerance"]
            value = value["target_value"]

        if approx(value, abs=tolerance) != saved_value:
            return (
                f"Saved value for metric '{metric_key}' ({saved_value}) does not match the requested "
                f"value ({value}) within requested tolerance ({tolerance})."
            )

        return None

    for metric_key in metrics_to_assert:
        if metric_key not in metrics_saved:
            errors.append(f"Metric '{metric_key}' not found in saved metrics.")
            continue

        value_to_assert = metrics_to_assert[metric_key]

        if isinstance(value_to_assert, dict):
            if "target_value" not in value_to_assert and "custom_tolerance" not in value_to_assert:
                # if it's a dictionary, call this function recursively
                # except when the dictionary has "target_value" and "custom_tolerance", which should
                # be treated as a regular dictionary
                errors.extend(_assert_metrics_dict(value_to_assert, metrics_saved[metric_key]))
                continue

        if isinstance(value_to_assert, list) and len(value_to_assert) > 0:
            # if it's a list, call an assertion for each element of the list
            for i in range(len(value_to_assert)):
                error = _assert(value_to_assert[i], metrics_saved[metric_key][i])
                if error is not None:
                    errors.append(error)
            continue

        # if it's just a regular value, perform the assertion
        error = _assert(value_to_assert, metrics_saved[metric_key])
        if error is not None:
            errors.append(error)

    return errors


def clear_metrics_folder() -> None:
    DEFAULT_METRICS_FOLDER.mkdir(exist_ok=True)
    for f in DEFAULT_METRICS_FOLDER.iterdir():
        if f.is_file() and str(f).endswith(".json"):
            f.unlink()


def load_metrics_from_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fedprox_example.server",
            client_python_path="examples.fedprox_example.client",
            config_path="tests/smoke_tests/fedprox_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            seed=42,
            server_metrics=load_metrics_from_file("tests/smoke_tests/fedprox_server_metrics.json"),
            client_metrics=load_metrics_from_file("tests/smoke_tests/fedprox_client_metrics.json"),
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.scaffold_example.server",
            client_python_path="examples.scaffold_example.client",
            config_path="tests/smoke_tests/scaffold_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            seed=42,
            server_metrics=load_metrics_from_file("tests/smoke_tests/scaffold_server_metrics.json"),
            client_metrics=load_metrics_from_file("tests/smoke_tests/scaffold_client_metrics.json"),
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.apfl_example.server",
            client_python_path="examples.apfl_example.client",
            config_path="tests/smoke_tests/apfl_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            seed=42,
            server_metrics=load_metrics_from_file("tests/smoke_tests/apfl_server_metrics.json"),
            client_metrics=load_metrics_from_file("tests/smoke_tests/apfl_client_metrics.json"),
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.feddg_ga_example.server",
            client_python_path="examples.feddg_ga_example.client",
            config_path="tests/smoke_tests/feddg_ga_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
            seed=42,
            server_metrics=load_metrics_from_file("tests/smoke_tests/feddg_ga_server_metrics.json"),
            client_metrics=load_metrics_from_file("tests/smoke_tests/feddg_ga_client_metrics.json"),
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.basic_example.server",
            client_python_path="examples.basic_example.client",
            config_path="tests/smoke_tests/basic_config.yaml",
            dataset_path="examples/datasets/cifar_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.dp_fed_examples.client_level_dp.server",
            client_python_path="examples.dp_fed_examples.client_level_dp.client",
            config_path="tests/smoke_tests/client_level_dp_config.yaml",
            dataset_path="examples/datasets/cifar_data/",
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
            dataset_path="examples/datasets/cifar_data/",
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
            dataset_path="examples/datasets/cifar_data/",
            checkpoint_path="examples/assets/best_checkpoint_fczjmljm.pkl",
            assert_evaluation_logs=True,
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fedper_example.server",
            client_python_path="examples.fedper_example.client",
            config_path="tests/smoke_tests/fedper_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fenda_example.server",
            client_python_path="examples.fenda_example.client",
            config_path="tests/smoke_tests/fenda_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.fl_plus_local_ft_example.server",
            client_python_path="examples.fl_plus_local_ft_example.client",
            config_path="tests/smoke_tests/fl_plus_local_ft_config.yaml",
            dataset_path="examples/datasets/cifar_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.moon_example.server",
            client_python_path="examples.moon_example.client",
            config_path="tests/smoke_tests/moon_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.run_until_complete(
        run_smoke_test(
            server_python_path="examples.ensemble_example.server",
            client_python_path="examples.ensemble_example.client",
            config_path="tests/smoke_tests/ensemble_config.yaml",
            dataset_path="examples/datasets/mnist_data/",
        )
    )
    loop.close()
