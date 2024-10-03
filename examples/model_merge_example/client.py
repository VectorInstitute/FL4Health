import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.clients.model_merge_client import ModelMergeClient
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import load_mnist_test_data
from fl4health.utils.metrics import Accuracy


class MnistModelMergeClient(ModelMergeClient):
    def get_model(self, config: Config) -> nn.Module:
        model = MnistNet()
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)

    def get_test_data_loader(self, config: Config) -> DataLoader:
        batch_size = narrow_dict_type(config, "batch_size", int)
        return load_mnist_test_data(self.data_path, batch_size)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        help="Path to the local dataset",
        default="examples/datasets/MNIST",
    )
    parser.add_argument("--model_path", action="store", type=str, help="Path to the clients checkpointed model")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client = MnistModelMergeClient(Path(args.dataset_path), Path(args.model_path), [Accuracy("acc")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
