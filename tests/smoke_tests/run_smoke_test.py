import asyncio
import logging

from six.moves import urllib

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()

# Working around mnist download issue
# https://github.com/pytorch/vision/issues/1938
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


async def run_smoke_test(
    n_clients_to_start: int = 4,
    config_path: str = "tests/smoke_tests/config.yaml",
    dataset_path: str = "examples/datasets/mnist_data/",
) -> None:
    # Start the server, divert the outputs to a server file
    logger.info("Starting server")

    server_process = await asyncio.create_subprocess_exec(
        "nohup",
        "python",
        "-m",
        "examples.fedprox_example.server",
        "--config_path",
        config_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    # reads lines from the server output in search of the startup log message
    # times out after 20s of inactivity if it doesn't find the log message
    full_server_output = ""
    startup_message = "FL starting"
    output_found = False
    while True:
        try:
            assert server_process.stdout is not None, "Server's process stdout is None"
            server_output_in_bytes = await asyncio.wait_for(server_process.stdout.readline(), 20)
            server_output = server_output_in_bytes.decode()
            full_server_output += server_output
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for server startup message '{startup_message}'")
            break
        logger.debug(f"Server output: {server_output}")
        if startup_message in server_output:
            output_found = True
            break

    assert output_found, f"Startup log message '{startup_message}' not found in server output."

    logger.info("Server started")

    # Start n number of clients and capture their process objects
    client_processes = []
    for i in range(n_clients_to_start):
        logger.info(f"Starting client {i}")
        client_process = await asyncio.create_subprocess_exec(
            "nohup",
            "python",
            "-m",
            "examples.fedprox_example.client",
            "--dataset_path",
            dataset_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        client_processes.append(client_process)

    # Collecting the clients output when their processes finish
    full_client_outputs = [""] * n_clients_to_start
    for i in range(len(client_processes)):
        logger.info(f"Waiting for client {i} to finish execution to collect its output...")
        client_output_bytes, client_err_bytes = await client_processes[i].communicate()
        logger.info(f"Output collected for client {i}")

        client_output = client_output_bytes.decode().replace("\\n", "\n")
        full_client_outputs[i] = client_output
        logger.debug(f"Client {i} stdout: {client_output}")

        if client_err_bytes is not None and len(client_err_bytes) > 0:
            client_err = client_err_bytes.decode().replace("\\n", "\n")
            full_client_outputs[i] += client_err
            logger.error(f"Client {i} stderr: {client_err}")

        # checking for clients with failure exit codes
        client_return_code = client_processes[i].returncode
        assert client_return_code is None or (client_return_code is not None and client_return_code == 0), (
            f"Full client output:\n{full_client_outputs[i]}\n"
            + f"[ASSERT ERROR] Client {i} exited with code {client_return_code}."
        )

    logger.info("All clients finished execution")

    logger.info("Waiting for the server to finish execution to collect its output...")
    server_output_bytes, server_err_bytes = await server_process.communicate()
    logger.info("Output collected for server")

    server_output = server_output_bytes.decode().replace("\\n", "\n")
    full_server_output = server_output
    logger.debug(f"Server stdout: {server_output}")

    if server_err_bytes is not None and len(server_err_bytes) > 0:
        server_err = server_err_bytes.decode().replace("\\n", "\n")
        full_server_output += server_err
        logger.error(f"Server stderr: {server_err}")

    # checking for clients with failure exit codes
    server_return_code = server_process.returncode
    assert server_return_code is None or (
        server_return_code is not None and server_return_code == 0
    ), f"Full output:\n{full_server_output}\n[ASSERT ERROR] Server exited with code {server_return_code}"

    logger.info("Server has finished execution")

    # server assertions
    assert (
        "error" not in full_server_output.lower()
    ), f"Full output:\n{full_server_output}\n[ASSERT ERROR] Error message found for server."
    # TODO pull this number from the config
    assert (
        "evaluate_round 15" in full_server_output
    ), f"Full output:\n{full_server_output}\n[ASSERT ERROR] Last FL round message not found for server."
    assert (
        "FL finished" in full_server_output
    ), f"Full output:\n{full_server_output}\n[ASSERT ERROR] FL finished message not found for server."
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
        assert (
            "error" not in full_client_outputs[i].lower()
        ), f"Full client output:\n{full_client_outputs[i]}\n[ASSERT ERROR] Error message found for client {i}."
        # TODO pull this number from the config
        assert "Current FL Round: 15" in full_client_outputs[i], (
            f"Full client output:\n{full_client_outputs[i]}\n"
            + f"[ASSERT ERROR] Last FL round message not found for client {i}."
        )
        assert "Disconnect and shut down" in full_client_outputs[i], (
            f"Full client output:\n{full_client_outputs[i]}\n"
            + f"[ASSERT ERROR] Shutdown message not found for client {i}."
        )


logger.info("All checks passed. Test finished.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_smoke_test())
    loop.close()
