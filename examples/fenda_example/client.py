import argparse
from pathlib import Path
from typing import Sequence, Set

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.models.fenda_cnn import FendaClassifier, GlobalCnn, LocalCnn
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.fenda_base import FendaJoinMode, FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import MinorityLabelBasedSampler


class MnistFendaClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        minority_numbers: Set[int],
        device: torch.device,
    ) -> None:
        super().__init__(data_path, metrics, device)
        self.minority_numbers = minority_numbers
        self.model = FendaModel(LocalCnn(), GlobalCnn(), FendaClassifier(FendaJoinMode.CONCATENATE)).to(self.device)
        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        downsample_percentage = self.narrow_config_type(config, "downsampling_ratio", float)

        sampler = MinorityLabelBasedSampler(list(range(10)), downsample_percentage, self.minority_numbers)

        train_loader, validation_loader, num_examples = load_mnist_data(self.data_path, batch_size, sampler)

        self.train_loader = train_loader
        self.val_loader = validation_loader
        self.num_examples = num_examples

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)


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
    client = MnistFendaClient(data_path, [Accuracy("accuracy")], minority_numbers, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
