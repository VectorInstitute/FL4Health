import argparse
from pathlib import Path
from typing import List, Set

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.datasets.dataset_utils import load_mnist_data
from examples.models.cnn_model import MNISTNet
from fl4health.clients.apfl_client import APFLClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.metrics import Accuracy, Metric


class MnistAPFLClient(APFLClient):
    def __init__(
        self,
        data_path: Path,
        minority_numbers: Set[int],
        metrics: List[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path=data_path, minority_numbers=minority_numbers, metrics=metrics, device=device)

    def setup_client(self, config: Config) -> None:
        downsampling_ratio = self.narrow_config_type(config, "downsampling_ratio", float)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        self.model: APFLModule = APFLModule(MNISTNet()).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.Adam(self.model.local_model.parameters(), lr=0.01)
        self.global_optimizer = torch.optim.Adam(self.model.global_model.parameters(), lr=0.01)
        self.train_loader, self.val_loader, self.num_examples = load_mnist_data(
            self.data_path, batch_size, downsampling_ratio, self.minority_numbers
        )
        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

        super().setup_client(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--minority_numbers", default=[], nargs="*", help="MNIST numbers to be in the minority for the current client"
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    minority_numbers = {int(number) for number in args.minority_numbers}

    model_constructor = MNISTNet

    client = MnistAPFLClient(data_path, minority_numbers, [Accuracy()], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
