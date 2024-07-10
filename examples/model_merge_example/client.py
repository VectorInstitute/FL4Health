import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.model_merge_client import ModelMergeClient
from fl4health.utils.load_data import load_mnist_test_data


class MnistModelMergeClient(ModelMergeClient):
    def get_model(self, config: Config) -> nn.Module:
        return MnistNet().to(self.device)

    def get_test_dataloader(self, config: Config) -> DataLoader:
        return load_mnist_test_data(self.data_path, 32)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        help="Path to the local dataset",
        default="examples/datasets/models/MNIST",
    )
    parser.add_argument("--model_path", action="store", type=str, help="Path to the clients checkpointed model")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client = MnistModelMergeClient(Path(args.dataset_path), Path(args.model_path), [], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
