import asyncio
import logging
from asyncio.subprocess import PIPE, STDOUT

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()


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
        stdout=PIPE,
        stderr=STDOUT,
    )

    # reads lines from the server output in search of the startup log message
    # times out after 20s of inactivity if it doesn't find the log message
    full_server_output = ""
    startup_message = "FL starting"
    output_found = False
    while True:
        try:
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
            stdout=PIPE,
            stderr=STDOUT,
        )
        client_processes.append(client_process)

    # Collecting the clients output until their processes finish
    full_client_outputs = [""] * n_clients_to_start
    # Clients that have finished execution are set to None, so the loop finishes when all of them are None
    # or in other words, while there are still any valid process objects in the list
    while any(client_processes):
        for i in range(len(client_processes)):
            if client_processes[i] is None:
                # Clients that have finished execution are set to None
                continue

            client_output_in_bytes = await asyncio.wait_for(client_processes[i].stdout.readline(), 20)
            client_output = client_output_in_bytes.decode()
            logger.debug(f"Client {i} output: {client_output}")

            full_client_outputs[i] += client_output

            # checking for clients with failure exit codes
            client_return_code = client_processes[i].returncode
            assert client_return_code is None or (client_return_code is not None and client_return_code == 0), \
                f"Client {i} exited with code {client_return_code}"

            if client_output is None or len(client_output) == 0 or client_return_code == 0:
                logger.info(f"Client {i} finished execution")
                # Setting the client that has finished to None
                client_processes[i] = None

    logger.info("All clients finished execution")

    # now wait for the server to finish
    while True:
        try:
            server_output_in_bytes = await asyncio.wait_for(server_process.stdout.readline(), 20)
            server_output = server_output_in_bytes.decode()
            full_server_output += server_output
            logger.debug(f"Server output: {server_output}")
        except asyncio.TimeoutError:
            logger.debug(f"Server log message retrieval timed out, it has likely finished execution")
            break

        # checking for clients with failure exit codes
        server_return_code = server_process.returncode
        assert server_return_code is None or (server_return_code is not None and server_return_code == 0), \
            f"Server exited with code {server_return_code}"

        if server_output is None or len(server_output) == 0 or server_return_code == 0:
            break

    logger.info("Server has finished execution")

    assert "error" not in full_server_output.lower(), "Error message found for server"
    for i in range(full_client_outputs):
        assert "error" not in full_client_outputs[i].lower(), f"Error message found for client {i}"
        # TODO pull this number from the config
        assert "Current FL Round: 15" in full_client_outputs[i], f"Last FL round message not found for client {i}"
        assert "Disconnect and shut down" in full_client_outputs[i], f"Shutdown message not found for client {i}"


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_smoke_test())
    loop.close()
