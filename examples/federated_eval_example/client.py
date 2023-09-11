import argparse
from pathlib import Path
from typing import Optional, Sequence

import flwr as fl
import torch
from flwr.common.typing import Config

from examples.models.cnn_model import Net
from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.utils.load_data import load_cifar10_test_data
from fl4health.utils.metrics import Accuracy, Metric


class CifarClient(EvaluateClient):
    def __init__(
        self, data_path: Path, metrics: Sequence[Metric], device: torch.device, model_checkpoint_path: Optional[Path]
    ) -> None:
        super().__init__(data_path, metrics, device, model_checkpoint_path)

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        # Initialized a global model to be hydrated with a server-side model if the parameters are passed
        self.global_model = Net().to(self.device)

        batch_size = self.narrow_config_type(config, "batch_size", int)

        evaluation_loader, num_examples = load_cifar10_test_data(self.data_path, batch_size)
        self.data_loader = evaluation_loader
        self.num_examples = num_examples

        self.criterion = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--checkpoint_path",
        action="store",
        type=str,
        help="Path to client model checkpoint.",
        required=False,
    )
    args = parser.parse_args()
    data_path = Path(args.dataset_path)
    client_checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    client = CifarClient(data_path, [Accuracy("accuracy")], DEVICE, client_checkpoint_path)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
