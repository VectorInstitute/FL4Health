import argparse
import os
from logging import INFO
from pathlib import Path
from typing import Dict, Sequence, Tuple

import flwr as fl
import torch
from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, BaselineLoss, FedIsic2019
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader, random_split

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.clients.apfl_client import ApflClient
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.utils.metrics import AccumulationMeter, BalancedAccuracy, Metric
from research.flamby.fed_isic2019.apfl.apfl_model import APFLEfficientNet


class FedIsic2019ApflClient(ApflClient):
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
        super().__init__(data_path=Path(""), metrics=metrics, device=device)
        assert 0 <= client_number < NUM_CLIENTS
        self.client_number = client_number
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.client_number}_best_model.pkl"
        self.learning_rate = learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        self.dataset_dir = dataset_dir

    def construct_train_val_datasets(self) -> Tuple[FedIsic2019, FedIsic2019]:
        full_train_dataset = FedIsic2019(
            center=self.client_number, train=True, pooled=False, data_path=self.dataset_dir
        )
        # Something weird is happening with the typing of the split sequence in random split. Punting with a mypy
        # ignore for now.
        train_dataset, validation_dataset = tuple(random_split(full_train_dataset, [0.8, 0.2]))  # type: ignore
        return train_dataset, validation_dataset

    def setup_client(self, config: Config) -> None:
        train_dataset, validation_dataset = self.construct_train_val_datasets()

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.num_examples = {"train_set": len(train_dataset), "validation_set": len(validation_dataset)}

        # NOTE: The class weights specified by alpha in this baseline loss are precomputed based on the weights of
        # the pool dataset. This is a bit of cheating but FLamby does it in their paper.
        self.criterion = BaselineLoss()

        self.model: APFLModule = APFLModule(APFLEfficientNet(), alpha_lr=self.alpha_learning_rate).to(self.device)
        self.local_optimizer = torch.optim.AdamW(self.model.local_model.parameters(), lr=self.learning_rate)
        self.global_optimizer = torch.optim.AdamW(self.model.global_model.parameters(), lr=self.learning_rate)

        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

        super().setup_client(config)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        global_meter = AccumulationMeter(self.metrics, "train_global")
        local_meter = AccumulationMeter(self.metrics, "train_local")
        personal_meter = AccumulationMeter(self.metrics, "train_personal")
        self.set_parameters(parameters, config)
        local_steps = self.narrow_config_type(config, "local_steps", int)
        metric_values = self.train_by_steps(local_steps, global_meter, local_meter, personal_meter)
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            metric_values,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        global_meter = AccumulationMeter(self.metrics, "val_global")
        local_meter = AccumulationMeter(self.metrics, "val_local")
        personal_meter = AccumulationMeter(self.metrics, "val_personal")
        loss, metric_values = self.validate(global_meter, local_meter, personal_meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )


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
    parser.add_argument(
        "--alpha_learning_rate", action="store", type=float, help="Learning rate for the APFL alpha", default=0.01
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")
    log(INFO, f"Learning Rate: {args.learning_rate}")
    log(INFO, f"Alpha Learning Rate: {args.alpha_learning_rate}")

    client = FedIsic2019ApflClient(
        args.learning_rate,
        args.alpha_learning_rate,
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
