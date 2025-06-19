import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common import Config
from torch import Tensor
from torch.utils.data import DataLoader

from fl4health.clients.fed_pca_client import FedPCAClient
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedPCAClient(FedPCAClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.5, beta=0.5)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_data_tensor(self, data_loader: DataLoader) -> Tensor:
        return torch.cat([inputs for inputs, _ in data_loader], dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset.")
    parser.add_argument(
        "--components_save_dir", action="store", type=str, help="Dir to which to save merged principal components."
    )
    parser.add_argument("--seed", action="store", type=int, help="Random seed for this client.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    components_save_dir = Path(args.components_save_dir)
    seed = args.seed

    # If the user wants to ensure that this example uses the same data as
    # the data used in the dim_reduction example, then both examples
    # should use the same random seed.
    set_all_random_seeds(seed)
    client = MnistFedPCAClient(data_path=data_path, device=device, model_save_dir=components_save_dir)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
