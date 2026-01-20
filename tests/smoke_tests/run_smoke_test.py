import asyncio
import contextlib
import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from flwr.common.typing import Config
from pytest import approx
from six.moves import urllib

from fl4health.utils.load_data import load_cifar10_data, load_mnist_data


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

DEFAULT_TOLERANCE = 0.0005
DEFAULT_READ_LOGS_TIMEOUT = 300

SERVER_STARTUP_MESSAGES = [
    # printed by fedprox, apfl, basic_example, fedbn, fedper, fedrep, and ditto, FENDA, fl_plus_local_ft & moon
    # Update this is no longer in output, examples are actually triggered by the [ROUND 1] startup message
    "FL starting",
    # printed by scaffold
    "Using Warm Start Strategy. Waiting for clients to be available for polling",
    # printed by client_level_dp, client_level_dp_weighted, instance_level_dp and dp_scaffold
    "Polling Clients for sample counts",
    # printed by federated_eval
    "Federated Evaluation Starting",
    "[ROUND 1]",
    # As far as I can tell this is printed by most servers that inherit from FlServer
    "Flower ECE: gRPC server running ",
    "gRPC server running",
    "server running",
]


# Custom Errors
class SmokeTestAssertError(Exception):
    pass


class SmokeTestExecutionError(Exception):
    pass


class SmokeTestTimeoutError(Exception):
    pass


def postprocess_logs(logs: str) -> str:
    """
    Postprocess logs to remove spurious Errors.

    This function removes a spurious 'error' that is sometimes thrown when
    running smoke tests on certain machines from the tcp_posix.cc library.
    It has been heavily investigated and doesn't not appear to actually affect
    FL processes.
    """
    return re.sub(r"E.*recvmsg encountered uncommon error: Message too long\n", "\n", logs)


def contains_actual_error(logs: str) -> bool:
    """
    Check if logs contain actual errors (exceptions, tracebacks) rather than just warnings.

    This function looks for patterns that indicate real errors such as Python tracebacks,
    exception messages, or gRPC errors, while ignoring deprecation warnings and other
    informational messages that happen to contain the word "error".

    Args:
        logs: The log output to check

    Returns:
        True if actual errors are found, False otherwise
    """
    # Check for Python exceptions and tracebacks
    if re.search(r"Traceback \(most recent call last\)", logs):
        return True

    # Check for common Python exception patterns (Error at end of line with context)
    # This matches lines like "ValueError: invalid literal" or "RuntimeError: connection failed"
    if re.search(r"^[A-Z]\w*Error:\s+\S", logs, re.MULTILINE):
        return True

    # Check for gRPC/network errors that indicate actual failures
    if re.search(r"(grpc\._channel\._InactiveRpcError|StatusCode\.[A-Z_]+)", logs):
        return True

    # Check for explicit ERROR log level messages (but not in DeprecationWarning)
    return bool(re.search(r"^ERROR:", logs, re.MULTILINE))


def graceful_shutdown(processes: list[asyncio.subprocess.Process]) -> None:
    """Graceful shutdown of subprocesses."""
    for p in processes:
        with contextlib.suppress(ProcessLookupError):
            p.terminate()


async def start_server_process_and_wait_for_stabilization(server_process: asyncio.subprocess.Process) -> str:
    full_server_output = ""

    output_found = False
    while not output_found:
        try:
            if not (server_process.stdout is not None):
                raise SmokeTestExecutionError("Server's process stdout is None")
            server_output_in_bytes = await asyncio.wait_for(server_process.stdout.readline(), 20)
            server_output = server_output_in_bytes.decode()
            logger.info(f"Server output: {server_output}")
            full_server_output += server_output
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for server startup messages")
            break

        return_code = server_process.returncode
        if return_code is not None and return_code != 0:
            raise SmokeTestAssertError(f"[ASSERT ERROR] Server exited with code {return_code}.")

        if any(startup_message in server_output for startup_message in SERVER_STARTUP_MESSAGES):
            output_found = True

    if not output_found:
        raise SmokeTestAssertError("[ASSERT_ERROR] Startup log message not found in server output.")

    logger.info("Server started")
    return full_server_output


async def run_smoke_test(
    server_python_path: str,
    client_python_path: str,
    config_path: str,
    dataset_path: str,
    checkpoint_path: str | None = None,
    assert_evaluation_logs: bool | None = False,
    additional_client_args: dict[str, Any] | None = None,
    # The param below exists to work around an issue with some clients
    # not printing the "Current FL Round" log message reliably
    skip_assert_client_fl_rounds: bool | None = False,
    seed: int | None = None,
    server_metrics: dict[str, Any] | None = None,
    client_metrics: dict[str, Any] | None = None,
    # assertion params
    tolerance: float = DEFAULT_TOLERANCE,
    read_logs_timeout: int = DEFAULT_READ_LOGS_TIMEOUT,
) -> tuple[list[str], list[str]]:
    """
    Runs a smoke test for a given server, client, and dataset configuration.

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
                checkpoint_path="examples/assets/fed_eval_example/best_checkpoint_global.pkl",
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
        server_python_path (str): The path for the executable server module.
        client_python_path (str): The path for the executable client module.
        config_path (str): The path for the config yaml file. The following attributes are required by this function:

            - `n_clients`: the number of clients to be started
            - `n_server_rounds`:  the number of rounds to be ran by the server
            - `batch_size`: the size of the batch, to be used by the dataset preloader

        partial_config_path (str): The path for the partial config yaml file. Used to pass to server and client to
            partially execute an FL run to validate checkpointing. The following attributes are required by this
            function and should be the same as those at config_path except n_server_rounds which should be 1:

            - `n_clients`: the number of clients to be started
            - `n_server_rounds`:  the number of rounds to be ran by the server
            - `batch_size`: the size of the batch, to be used by the dataset preloader

        dataset_path (str): The path of the dataset. Depending on which dataset is being used, it will ty to preload it
            to avoid problems when running on different runtimes.
        checkpoint_path (str | None, optional): If set, it will send that path as a model checkpoint path
            to the client. Defaults to None.
        assert_evaluation_logs (bool | None, optional): Set this to `True` if testing an evaluation model, which
            produces different log outputs. Defaults to False.
        additional_client_args (dict[str, Any] | None): Additional command line arguments for client command. Defaults
            to None.
        skip_assert_client_fl_rounds (bool | None, optional): If set to `True`, will skip the assertion of the
            "Current FL Round" message on the clients' logs. This is necessary because some clients (namely
            ``client_level_dp``, ``client_level_dp_weighted``, ``instance_level_dp``) do not reliably print that
            message. Defaults to False.
        seed (int | None, optional): The random seed to be passed in to both the client and the server.
            Defaults to None.
        server_metrics (dict[str, Any] | None, optional): A dictionary of metrics to be checked against the metrics
            file saved by the server. Should be in the same format as ``fl4health.reporting.metrics.MetricsReporter``.
            Defaults to None.
        client_metrics (dict[str, Any] | None, optional): A dictionary of metrics to be checked against the metrics
            file saved by the clients. Should be in the same format as ``fl4health.reporting.metrics.MetricsReporter.``
            Defaults to None.
        tolerance (float, optional): Tolerance associated with natural metrics fluctuations. Defaults to
            DEFAULT_TOLERANCE.
        read_logs_timeout (int, optional):Timeout in seconds associated with reading the logs of the clients and
            servers. Defaults to DEFAULT_READ_LOGS_TIMEOUT

    Raises:
        SmokeTestAssertError: Various assertions are raised when metrics do not match or the clients/server processes
            do not finish correctly

    Returns:
        (tuple[list[str], list[str]]): (server_errors, client_errors) list of errors from server and client processes,
            respectively.
    """
    try:
        clear_metrics_folder()

        logger.info("Running smoke tests with parameters:")
        logger.info(f"\tServer : {server_python_path}")
        logger.info(f"\tClient : {client_python_path}")
        logger.info(f"\tConfig : {config_path}")
        logger.info(f"\tDataset: {dataset_path}")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        _preload_dataset(dataset_path, config, seed)

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

        # Reads lines from the server output in search of the startup log message. Times out after 20s of inactivity
        # if it doesn't find the log message
        _ = await start_server_process_and_wait_for_stabilization(server_process)

        # Start n number of clients and capture their process objects
        client_tasks = []
        for i in range(config["n_clients"]):
            logger.info(f"Starting client {i}")

            client_args = ["-m", client_python_path, "--dataset_path", dataset_path]
            if checkpoint_path is not None:
                client_args.extend(["--checkpoint_path", checkpoint_path])
            if seed is not None:
                client_args.extend(["--seed", str(seed)])
            if additional_client_args:
                for k, v in additional_client_args.items():
                    client_args.extend([k, str(v)])

            task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    "python",
                    *client_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            )
            client_tasks.append(task)

        client_processes = await asyncio.gather(*client_tasks)

        # Collecting the clients output when their processes finish
        client_result_tasks = []
        for i, client_process in enumerate(client_processes):
            client_result_tasks.append(
                _wait_for_process_to_finish_and_retrieve_logs(client_process, f"Client {i}", read_logs_timeout),
            )

        full_client_outputs = await asyncio.gather(*client_result_tasks)
        logger.info("All clients finished execution")

        # Collecting the server output when its process finish
        full_server_output = await _wait_for_process_to_finish_and_retrieve_logs(
            server_process, "Server", read_logs_timeout
        )
        full_server_output = postprocess_logs(full_server_output)

        logger.info("Server has finished execution")

        # Server assertions
        if contains_actual_error(full_server_output):
            raise SmokeTestAssertError("[ASSERT ERROR] Error message found for server.")

        if assert_evaluation_logs:
            if f"Federated Evaluation received {config['n_clients']} results and 0 failures" not in full_server_output:
                raise SmokeTestAssertError("[ASSERT ERROR] Last FL round message not found for server.")

            if "Federated Evaluation Finished" not in full_server_output:
                raise SmokeTestAssertError(
                    "[ASSERT ERROR] Federated Evaluation Finished message not found for server."
                )
            if not (all(message in full_server_output for message in ["History (metrics, distributed, evaluate):"])):
                raise SmokeTestAssertError("[ASSERT ERROR] Metrics message not found for server.")
        else:
            if "[SUMMARY]" not in full_server_output:
                raise SmokeTestAssertError("[ASSERT ERROR] [SUMMARY] message not found for server.")

            if not (
                all(
                    message in full_server_output
                    for message in [
                        "History (loss, distributed):",
                        "History (metrics, distributed, fit):",
                    ]
                )
            ):
                raise SmokeTestAssertError("[ASSERT ERROR] Metrics message not found for server.")

        server_errors = _assert_metrics(MetricType.SERVER, server_metrics, tolerance)

        # Client assertions
        client_errors = []
        for i, raw_full_client_output in enumerate(full_client_outputs):
            full_client_output = postprocess_logs(raw_full_client_output)

            if contains_actual_error(full_client_output):
                raise SmokeTestAssertError(f"[ASSERT ERROR] Error message found for client {i}.")

            if "Disconnect and shut down" not in full_client_output:
                raise SmokeTestAssertError(f"[ASSERT ERROR] Shutdown message not found for client {i}.")

            if assert_evaluation_logs:
                if "Client Evaluation Local Model Metrics" not in full_client_output:
                    raise SmokeTestAssertError(
                        f"[ASSERT ERROR] 'Client Evaluation Local Model Metrics' message not found for client {i}."
                    )
            elif (
                not skip_assert_client_fl_rounds
                and f"Current FL Round: {config['n_server_rounds']}" not in full_client_output
            ):
                raise SmokeTestAssertError(f"[ASSERT ERROR] Last FL round message not found for client {i}.")

            client_errors.extend(_assert_metrics(MetricType.CLIENT, client_metrics, tolerance))

        return server_errors, client_errors
    except Exception:
        raise
    finally:
        if server_process:
            graceful_shutdown([server_process] + client_processes)
        else:
            graceful_shutdown(client_processes)


async def run_fault_tolerance_smoke_test(
    server_python_path: str,
    client_python_path: str,
    config_path: str,
    partial_config_path: str,
    dataset_path: str,
    server_metrics: dict[str, Any],
    client_metrics: dict[str, Any],
    seed: int | None = None,
    intermediate_checkpoint_dir: str = "./",
    server_name: str = "server",
    tolerance: float = DEFAULT_TOLERANCE,
    read_logs_timeout: int = DEFAULT_READ_LOGS_TIMEOUT,
) -> tuple[list[str], list[str]]:
    """
    Runs a smoke test for a given server, client, and dataset configuration.

    Args:
        server_python_path (str): The path for the executable server module.
        client_python_path (str): The path for the executable client module.
        config_path (str): The path for the config yaml file. The following attributes are required by this function:

            - `n_clients`: the number of clients to be started
            - `n_server_rounds`:  the number of rounds to be ran by the server
            - `batch_size`: the size of the batch, to be used by the dataset preloader

        partial_config_path (str): The path for the partial config yaml file. Used to pass to server and client to
            partially execute an FL run to validate checkpointing. The following attributes are required by this
            function and should be the same as those at config_path except n_server_rounds which should be 1:

            - `n_clients`: the number of clients to be started
            - `n_server_rounds`:  the number of rounds to be ran by the server
            - `batch_size`: the size of the batch, to be used by the dataset preloader

        dataset_path (str): The path of the dataset. Depending on which dataset is being used, it will ty to preload it
            to avoid problems when running on different runtimes.
        server_metrics (dict[str, Any]): A dictionary of metrics to be checked against the metrics file
            saved by the server. Should be in the same format as ``fl4health.reporting.metrics.MetricsReporter``.
        client_metrics (dict[str, Any]): A dictionary of metrics to be checked against the metrics file
            saved by the clients. Should be in the same format as ``fl4health.reporting.metrics.MetricsReporter.``
        seed (int | None, optional): The random seed to be passed in to both the client and the server.
            Defaults to None.
        intermediate_checkpoint_dir (str, optional): Path to store intermediate checkpoints for server and client.
            Defaults to "./".
        server_name (str, optional): Name of the server class. This is often used for naming checkpoints and other
            artifacts. Defaults to "server".
        tolerance (float, optional): Tolerance associated with natural metrics fluctuations. Defaults to
            DEFAULT_TOLERANCE.
        read_logs_timeout (int, optional): Timeout in seconds associated with reading the logs of the clients and
            servers. Defaults to DEFAULT_READ_LOGS_TIMEOUT.

    Returns:
        (tuple[list[str], list[str]]): (server_errors, client_errors) list of errors from server and client processes,
            respectively.
    """
    try:
        clear_metrics_folder()

        logger.info("Running smoke tests with parameters:")
        logger.info(f"\tServer : {server_python_path}")
        logger.info(f"\tClient : {client_python_path}")
        logger.info(f"\tConfig : {config_path}")
        logger.info(f"\tDataset: {dataset_path}")
        logger.info(f"\tCheckpoint Directory: {intermediate_checkpoint_dir}")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        _preload_dataset(dataset_path, config, seed)

        # Start the server and capture its process object
        logger.info("Starting server...")
        partial_server_args = [
            "-m",
            server_python_path,
            "--config_path",
            partial_config_path,
            "--intermediate_server_state_dir",
            intermediate_checkpoint_dir,
            "--server_name",
            server_name,
        ]
        server_args = [
            "-m",
            server_python_path,
            "--config_path",
            config_path,
            "--intermediate_server_state_dir",
            intermediate_checkpoint_dir,
            "--server_name",
            server_name,
        ]
        client_args = [
            "-m",
            client_python_path,
            "--dataset_path",
            dataset_path,
            "--intermediate_client_state_dir",
            intermediate_checkpoint_dir,
        ]
        if seed is not None:
            partial_server_args.extend(["--seed", str(seed)])
            server_args.extend(["--seed", str(seed)])
            client_args.extend(["--seed", str(seed)])

        server_process = await asyncio.create_subprocess_exec(
            "python",
            *partial_server_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Reads lines from the server output in search of the startup log message. Times out after 20s of inactivity
        # if it doesn't find the log message
        _ = await start_server_process_and_wait_for_stabilization(server_process)

        # Start n number of clients and capture their process objects
        client_tasks = []
        for i in range(config["n_clients"]):
            logger.info(f"Starting client {i}")

            curr_client_args = client_args + ["--client_name", str(i)]

            client_tasks.append(
                asyncio.create_subprocess_exec(
                    "python",
                    *curr_client_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            )

        client_processes = await asyncio.gather(*client_tasks)

        client_output_tasks = []
        for i in range(len(client_processes)):
            client_output_tasks.append(
                _wait_for_process_to_finish_and_retrieve_logs(client_processes[i], f"Client {i}", read_logs_timeout),
            )

        _ = await asyncio.gather(*client_output_tasks)

        logger.info("All clients finished execution")

        await _wait_for_process_to_finish_and_retrieve_logs(server_process, "Server", read_logs_timeout)

        logger.info("Server has finished execution")

        server_process = await asyncio.create_subprocess_exec(
            "python",
            *server_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        # Reads lines from the server output in search of the startup log message. Times out after 20s of inactivity
        # if it doesn't find the log message
        _ = await start_server_process_and_wait_for_stabilization(server_process)

        # Start n number of clients and capture their process objects
        client_processes = []
        for i in range(config["n_clients"]):
            logger.info(f"Starting client {i}")

            curr_client_args = client_args + ["--client_name", str(i)]

            client_process = await asyncio.create_subprocess_exec(
                "python",
                *curr_client_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            client_processes.append(client_process)

        for i in range(len(client_processes)):
            await _wait_for_process_to_finish_and_retrieve_logs(client_processes[i], f"Client {i}", read_logs_timeout)

        logger.info("All clients finished execution")

        await _wait_for_process_to_finish_and_retrieve_logs(server_process, "Server", read_logs_timeout)

        logger.info("Server has finished execution")

        server_errors = _assert_metrics(MetricType.SERVER, server_metrics, tolerance)

        # client assertions
        client_errors = []
        client_errors.extend(_assert_metrics(MetricType.CLIENT, client_metrics, tolerance))

        return server_errors, client_errors
    except Exception:
        raise
    finally:
        if server_process:
            graceful_shutdown([server_process] + client_processes)
        else:
            graceful_shutdown(client_processes)


def _preload_dataset(dataset_path: str, config: Config, seed: int | None = None) -> None:
    if "mnist" in dataset_path:
        logger.info("Preloading MNIST dataset...")

        # Working around MNIST download issue
        # https://github.com/pytorch/vision/issues/1938
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        # Creating a client and getting the data loaders will trigger
        # the dataset's download
        logger.info("Preloading MNIST dataset...")

        if seed is not None:
            load_mnist_data(Path(dataset_path), int(config["batch_size"]))
        else:
            load_mnist_data(Path(dataset_path), int(config["batch_size"]))

        logger.info("Finished preloading MNIST dataset")

    elif "cifar" in dataset_path:
        logger.info("Preloading CIFAR10 dataset...")

        if seed is not None:
            load_cifar10_data(Path(dataset_path), int(config["batch_size"]))
        else:
            load_cifar10_data(Path(dataset_path), int(config["batch_size"]))

        logger.info("Finished preloading CIFAR10 dataset")

    else:
        logger.info("Preload not supported for specified dataset. Skipping.")


async def _wait_for_process_to_finish_and_retrieve_logs(
    process: asyncio.subprocess.Process,
    process_name: str,
    timeout: int = 300,  # timeout for the whole process to complete
) -> str:
    async def get_output_from_stdout(stream_reader: asyncio.streams.StreamReader) -> tuple[str, int | None]:
        """
        Reading from the stdout stream.

        **NOTE**: In the initial implementation, we were invoking asyncio.wait_for()
        with the coroutine stream_reader.readline() and with a timeout=timeout.
        Outsourcing to this function now which we wrap this whole helper fn
        with asyncio.wait_for() for improved readability and better error
        handling, especially of the TimeoutError.
        """
        full_output = ""
        while True:
            output_in_bytes = await stream_reader.readline()
            await asyncio.sleep(0)  # give control back to loop manager
            output = output_in_bytes.decode().replace("\\n", "\n")
            logger.info(f"{process_name} output: {output}")
            full_output += output
            return_code = process.returncode

            if output == "" and return_code is not None:
                break

        return full_output, return_code

    logger.info(f"Collecting output for {process_name}...")

    try:
        if process.stdout is None:
            raise SmokeTestExecutionError("Process stdout is None")
        full_output, return_code = await asyncio.wait_for(get_output_from_stdout(process.stdout), timeout=timeout)
    except asyncio.exceptions.TimeoutError as e:
        raise SmokeTestTimeoutError(f"Timeout for reading logs of {process_name} reached.") from e
    except Exception as ex:
        logger.exception(f"Error collecting {process_name} log messages:")
        raise ex

    logger.info(f"Output collected for {process_name}")

    # checking for processes with failure exit codes
    if not (return_code is None or (return_code is not None and return_code == 0)):
        raise SmokeTestAssertError(f"[ASSERT ERROR] {process_name} exited with code {return_code}.")

    return full_output


class MetricType(Enum):
    SERVER = "server"
    CLIENT = "client"


DEFAULT_METRICS_FOLDER = Path("metrics")


def _assert_metrics(
    metric_type: MetricType, metrics_to_assert: dict[str, Any] | None = None, tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    errors: list[str] = []
    if metrics_to_assert is None:
        return errors

    metrics_found = False
    for file in DEFAULT_METRICS_FOLDER.iterdir():
        if not file.is_file() or not str(file).endswith(".json"):
            continue

        with open(file) as f:
            metrics = json.load(f)

        if metrics["host_type"] != metric_type.value:
            continue

        metrics_found = True
        errors.extend(_assert_metrics_dict(metrics_to_assert, metrics, tolerance))

    if not metrics_found:
        errors.append(f"Metrics of type {metric_type.value} not found.")

    return errors


def _assert_metrics_dict(
    metrics_to_assert: dict[str, Any], metrics_saved: dict[str, Any], tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    errors = []

    def _assert(value: Any, saved_value: Any, tolerance: float) -> str | None:
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

    for metric_key, metric_val in metrics_to_assert.items():
        if metric_key not in metrics_saved:
            errors.append(f"Metric '{metric_key}' not found in saved metrics.")
            continue

        value_to_assert = metric_val

        if (
            isinstance(value_to_assert, dict)
            and "target_value" not in value_to_assert
            and "custom_tolerance" not in value_to_assert
        ):
            # if it's a dictionary, call this function recursively
            # except when the dictionary has "target_value" and "custom_tolerance", which should
            # be treated as a regular dictionary
            errors.extend(_assert_metrics_dict(value_to_assert, metrics_saved[metric_key], tolerance))
            continue

        if isinstance(value_to_assert, list) and len(value_to_assert) > 0:
            # if it's a list, call an assertion for each element of the list
            for i in range(len(value_to_assert)):
                error = _assert(value_to_assert[i], metrics_saved[metric_key][i], tolerance)
                if error is not None:
                    errors.append(error)
            continue

        # if it's just a regular value, perform the assertion
        error = _assert(value_to_assert, metrics_saved[metric_key], tolerance)
        if error is not None:
            errors.append(error)

    return errors


def clear_metrics_folder() -> None:
    DEFAULT_METRICS_FOLDER.mkdir(exist_ok=True)
    for f in DEFAULT_METRICS_FOLDER.iterdir():
        if f.is_file() and str(f).endswith(".json"):
            f.unlink()


def load_metrics_from_file(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)
