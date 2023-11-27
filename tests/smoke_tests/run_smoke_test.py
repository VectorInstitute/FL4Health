import logging
import subprocess
import time

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def run_smoke_test(
    n_clients_to_start: int = 4,
    config_path: str = "tests/smoke_tests/config.yaml",
    dataset_path: str = "examples/datasets/mnist_data/",
) -> None:
    # Start the server, divert the outputs to a server file
    logger.info("Starting server")

    ps = subprocess.Popen([
        "nohup",
        "python",
        "-m",
        "examples.fedprox_example.server",
        "--config_path",
        config_path,
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    server_output = ps.stdout

    # Sleep for 20 seconds to allow the server to come up.
    # TODO fix this by capturing the output and parsing it
    time.sleep(20)

    # Start n number of clients and divert the outputs to their own files
    client_outputs = []
    for i in range(n_clients_to_start):
        logger.info(f"Starting client {i}")
        ps = subprocess.Popen([
            "nohup",
            "python",
            "-m",
            "examples.fedprox_example.client",
            "--dataset_path",
            dataset_path,
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        client_outputs.append(ps.stdout)

    # TODO make asserts

    while True:
        logger.info(f"Server output: {server_output.readline().decode()}")
        for i in range(len(client_outputs)):
            logger.info(f"Client {i} output: {client_outputs[i].readline().decode()}")


if __name__ == "__main__":
    run_smoke_test()
