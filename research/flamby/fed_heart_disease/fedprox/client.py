import argparse
from logging import INFO
from typing import Sequence

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.metrics import Accuracy, Metric
from research.flamby.flamby_clients.flamby_fedprox_client import FlambyFedProxClient
from research.flamby.flamby_data_utils import construct_fed_heard_disease_train_val_datasets


class FedHeartDiseaseFedProxClient(FlambyFedProxClient):
    def __init__(
        self,
        learning_rate: float,
        mu: float,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        assert 0 <= client_number < NUM_CLIENTS
        super().__init__(learning_rate, mu, metrics, device, client_number, checkpoint_stub, dataset_dir, run_name)

    def setup_client(self, config: Config) -> None:
        train_dataset, validation_dataset = construct_fed_heard_disease_train_val_datasets(
            self.client_number, self.dataset_dir
        )

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_examples = {"train_set": len(train_dataset), "validation_set": len(validation_dataset)}

        self.model: nn.Module = Baseline().to(self.device)

        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # Set the Proximal Loss weight mu
        self.proximal_weight = self.mu

        self.parameter_exchanger = FullParameterExchanger()

        super().setup_client(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Path to the preprocessed Fed Heart Disease Dataset (ex. path/to/fed_heart_disease)",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--client_number",
        action="store",
        type=int,
        help="Number of the client for dataset loading (should be 0-3 for Fed Heart Disease)",
        required=True,
    )
    parser.add_argument("--mu", action="store", type=float, help="Mu value for the FedProx training", default=0.1)
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"FedProx Mu: {args.mu}")

    client = FedHeartDiseaseFedProxClient(
        args.learning_rate,
        args.mu,
        [Accuracy("FedHeartDisease_accuracy")],
        DEVICE,
        args.client_number,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
