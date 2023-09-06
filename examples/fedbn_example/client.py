import argparse
from logging import INFO
from pathlib import Path
from typing import Sequence

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNetWithBnAndFrozen
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.layer_exchanger import LayerExchangerWithExclusions
from fl4health.reporting.fl_wanb import ClientWandBReporter
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedBNClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)
        log(INFO, f"Client Name: {self.client_name}")

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        self.model: nn.Module = MnistNetWithBnAndFrozen(freeze_cnn_layer=False).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=1)

        self.train_loader, self.val_loader, self.num_examples = load_mnist_data(self.data_path, batch_size, sampler)

        # All that we need to do on the client side to apply the FedBN approach is to use the
        # LayerExchangerWithExclusions and specify that we want to exclude the BatchNorm layers of our model
        self.parameter_exchanger = LayerExchangerWithExclusions(self.model, {nn.BatchNorm2d})

        # Setup W and B reporter
        self.wandb_reporter = ClientWandBReporter.from_config(self.client_name, config)

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

    client = MnistFedBNClient(data_path, [Accuracy()], DEVICE)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
