import argparse
from pathlib import Path
from typing import Sequence

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Accuracy, AverageMeter, Metric


class CifarClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device) -> None:
        super().__init__(data_path, metrics, device)
        self.model = Net().to(self.device)
        self.parameter_exchanger = FullParameterExchanger()

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)

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

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    metrics = [Accuracy("accuracy")]
    client = CifarClient(data_path, metrics, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

    # Run further local training after the federated learning has finished
    meter = AverageMeter(metrics, "train_meter")
    client.train_by_epochs(2, meter)
    # Finally, we evaluate the model
    client.validate(meter)
