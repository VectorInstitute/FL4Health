import argparse
from logging import INFO
from typing import Sequence

import flwr as fl
import torch
from flamby.datasets.fed_ixi import BATCH_SIZE, LR, NUM_CLIENTS, BaselineLoss
from flwr.common.logger import log
from flwr.common.typing import Config
from torch.utils.data import DataLoader

from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.metrics import BinarySoftDiceCoefficient, Metric
from research.flamby.fed_ixi.apfl.apfl_model import APFLUNet
from research.flamby.flamby_clients.flamby_apfl_client import FlambyApflClient
from research.flamby.flamby_data_utils import construct_fed_ixi_train_val_datasets


class FedIxiApflClient(FlambyApflClient):
    def __init__(
        self,
        learning_rate: float,
        alpha_learning_rate: float,
        metrics: Sequence[Metric],
        device: torch.device,
        client_number: int,
        checkpoint_stub: str,
        dataset_dir: str,
        run_name: str = "",
    ) -> None:
        assert 0 <= client_number < NUM_CLIENTS
        super().__init__(
            learning_rate, alpha_learning_rate, metrics, device, client_number, checkpoint_stub, dataset_dir, run_name
        )

    def setup_client(self, config: Config) -> None:
        train_dataset, validation_dataset = construct_fed_ixi_train_val_datasets(self.client_number, self.dataset_dir)

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_examples = {"train_set": len(train_dataset), "validation_set": len(validation_dataset)}

        self.criterion = BaselineLoss()

        self.model: APFLModule = APFLModule(APFLUNet(), alpha_lr=self.alpha_learning_rate).to(self.device)
        self.local_optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=self.learning_rate)
        self.global_optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=self.learning_rate)

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
        help="Path to the preprocessed FedIXI Dataset (ex. path/to/fedixi/dataset)",
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
        help="Number of the client for dataset loading (should be 0-2 for FedIXI)",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=LR
    )
    parser.add_argument(
        "--alpha_learning_rate", action="store", type=float, help="Learning rate for the APFL alpha", default=0.01
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Alpha Learning Rate: {args.alpha_learning_rate}")

    client = FedIxiApflClient(
        args.learning_rate,
        args.alpha_learning_rate,
        [BinarySoftDiceCoefficient("FedIXI_dice")],
        DEVICE,
        args.client_number,
        args.artifact_dir,
        args.dataset_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
