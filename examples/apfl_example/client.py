import argparse
from pathlib import Path
from typing import List

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNet
from fl4health.clients.apfl_client import ApflClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistApflClient(ApflClient):
    def __init__(
        self,
        data_path: Path,
        metrics: List[Metric],
        device: torch.device,
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        self.model: APFLModule = APFLModule(MnistNet()).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=0.01)
        self.global_optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=0.01)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75)

        self.train_loader, self.val_loader, self.num_examples = load_mnist_data(self.data_path, batch_size, sampler)
        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

        super().setup_client(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = MnistApflClient(data_path, [Accuracy()], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
