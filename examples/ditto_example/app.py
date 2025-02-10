# from flwr.simulation import run_simulation
# from flwr.server import ServerApp
# from flwr.client import ClientApp
# import argparse
# from logging import INFO
# from pathlib import Path

# import flwr as fl
# from flwr.common.logger import log
# from fl4health.utils.random import set_all_random_seeds
# from fl4health.reporting import JsonReporter

# import argparse
# import torch

# from examples.ditto_example.server_fn import server_fn
# from fl4health.utils.config import load_config
# from examples.ditto_example.client_fn import get_client_fn
# from functools import partial


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="FL Simulation Main")
#     parser.add_argument(
#         "--config_path",
#         action="store",
#         type=str,
#         help="Path to configuration file.",
#         default="examples/ditto_example/config.yaml",
#     )
#     parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
#     parser.add_argument(
#         "--server_address",
#         action="store",
#         type=str,
#         help="Server Address for the clients to communicate with the server through",
#         default="0.0.0.0:8080",
#     )
#     parser.add_argument(
#         "--seed",
#         action="store",
#         type=int,
#         help="Seed for the random number generators across python, torch, and numpy",
#         required=False,
#     )
#     args = parser.parse_args()

#     config = load_config(args.config_path)

#     server = server_fn(config)

    

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     data_path = Path(args.dataset_path)
#     log(INFO, f"Device to be used: {device}")
#     log(INFO, f"Server Address: {args.server_address}")

#     # Set the random seed for reproducibility
#     set_all_random_seeds(args.seed)

#     client_fn = partial(get_client_fn, data_path, device)
#     client_app = ClientApp(client_fn)

#     # Create your ServerApp passing the server generation function
#     server_app = ServerApp(server=server)

#     run_simulation(
#         server_app=server_app,
#         client_app=client_app,
#         num_supernodes=10,  # equivalent to setting `num-supernodes` in the pyproject.toml
#     )

