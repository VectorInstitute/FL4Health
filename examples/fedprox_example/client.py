import argparse
from logging import INFO
from pathlib import Path
from typing import List

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNet
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedProxClient(FedProxClient):
    def __init__(
        self,
        data_path: Path,
        metrics: List[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        self.model: nn.Module = MnistNet().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        # Set the Proximal Loss weight mu
        self.proximal_weight = 0.1
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)

        self.train_loader, self.val_loader, self.num_examples = load_mnist_data(self.data_path, batch_size, sampler)
        self.parameter_exchanger = FullParameterExchanger()

        super().setup_client(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    client = MnistFedProxClient(data_path, [Accuracy()], DEVICE)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
