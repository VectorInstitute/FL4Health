import argparse
from logging import INFO
from typing import Sequence

import flwr as fl
import torch
import torch.nn as nn
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.metrics import BalancedAccuracy, Metric
from research.flamby.fed_isic2019.fenda.fenda_model import FedIsic2019FendaModel
from research.flamby.flamby_clients.flamby_fedopt_client import FlambyFedOptClient
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets


class FedIsic2019FendaClient(FlambyFedOptClient):
    def __init__(
        self,
        learning_rate: float,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        assert 0 <= client_number < NUM_CLIENTS
        super().__init__(learning_rate, metrics, device, client_number, checkpoint_stub, dataset_dir, run_name)

    def setup_client(self, config: Config) -> None:
        train_dataset, validation_dataset = construct_fedisic_train_val_datasets(self.client_number, self.dataset_dir)

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_examples = {"train_set": len(train_dataset), "validation_set": len(validation_dataset)}

        self.model: nn.Module = FedIsic2019FendaModel(frozen_blocks=13, turn_off_bn_tracking=False).to(self.device)
        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = BaselineLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

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
        help="Path to the preprocessed FedIsic2019 Dataset (ex. path/to/fedisic2019)",
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
        help="Number of the client for dataset loading (should be 0-5 for FedIsic2019)",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")

    client = FedIsic2019FendaClient(
        args.learning_rate,
        [BalancedAccuracy("FedIsic2019_balanced_accuracy")],
        DEVICE,
        args.client_number,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
