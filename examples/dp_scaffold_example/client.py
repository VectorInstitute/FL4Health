import argparse
from pathlib import Path
from typing import Optional, Sequence

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config

from examples.models.cnn_model import MnistNet
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.scaffold_client import DPScaffoldClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistDPScaffoldClient(DPScaffoldClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        learning_rate_local: float,
        meter_type: str = "average",
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            learning_rate_local=learning_rate_local,
            meter_type=meter_type,
            use_wandb_reporter=use_wandb_reporter,
            checkpointer=checkpointer,
        )

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        learning_rate_local = self.narrow_config_type(config, "learning_rate_local", float)

        self.noise_multiplier = self.narrow_config_type(config, "noise_multiplier", float)
        self.clipping_bound = self.narrow_config_type(config, "clipping_bound", float)

        self.learning_rate_local = learning_rate_local
        self.model: nn.Module = MnistNet().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate_local)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=1.0)

        self.train_loader, self.val_loader, self.num_examples = load_mnist_data(self.data_path, batch_size, sampler)
        model_size = len(self.model.state_dict())
        self.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        self.setup_opacus_objects()
        super().setup_client(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")

    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = MnistDPScaffoldClient(data_path, [Accuracy()], DEVICE, 0.05)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
