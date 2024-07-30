import argparse
from pathlib import Path
from typing import Sequence

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import narrow_config_type
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy, Metric


class CifarClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device) -> None:
        super().__init__(data_path, metrics, device)
        self.model = Net()
        self.parameter_exchanger = FullParameterExchanger()

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = narrow_config_type(config, "batch_size", int)
        train_loader, validation_loader, num_examples = load_cifar10_data(self.data_path, batch_size)

        self.train_loader = train_loader
        self.val_loader = validation_loader
        self.num_examples = num_examples

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CifarClient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_client(server_address="fl_server:8080", client=client.to_client())
