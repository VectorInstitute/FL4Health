import argparse
from pathlib import Path
from typing import Tuple

import flwr as fl
import torch
from flwr.common import Config
from torch import Tensor
from torch.utils.data import DataLoader

from fl4health.clients.fed_pca_client import FedPCAClient
from fl4health.utils.config import narrow_config_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedPCAClient(FedPCAClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.5, beta=0.5)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_data_tensor(self, data_loader: DataLoader) -> Tensor:
        all_data_tensor = torch.cat([inputs for inputs, _ in data_loader], dim=0)
        return all_data_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset.")
    parser.add_argument(
        "--components_save_path", action="store", type=str, help="Path to saving merged principal components."
    )
    parser.add_argument("--seed", action="store", type=int, help="Random seed for this client.")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    components_save_path = Path(args.components_save_path)
    seed = args.seed

    # If the user wants to ensure that this example uses the same data as
    # the data used in the dim_reduction example, then both examples
    # should use the same random seed.
    set_all_random_seeds(seed)
    client = MnistFedPCAClient(data_path=data_path, device=DEVICE, model_save_path=components_save_path)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
