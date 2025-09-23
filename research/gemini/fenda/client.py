import argparse
import os
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from data.data import load_train_delirium, load_train_mortality
from fl4health.clients.numpy_fl_client import NumpyFlClient
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.metrics import AccumulationMeter, Meter, Metric

# FENDA imports
from fl4health.model_bases.fenda_base import FendaJoinMode, FendaModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger

# delirium model
from research.gemini.delirium_models.fenda_mlp import FendaClassifierD, GlobalMlpD, LocalMlpD
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc

# mortality model
from research.gemini.mortality_models.fenda_mlp import FendaClassifier, GlobalMLP, LocalMLP


class GeminiFendaClient(NumpyFlClient):
    def __init__(
        self,
        data_path: Path,
        metrics: list[Metric],
        hospitals_id: list[str],
        device: torch.device,
        learning_task: str,
        learning_rate: float,
        checkpoint_stub: str,
        run_name: str = "",
    ) -> None:
        super().__init__(data_path=data_path, device=device)
        self.hospitals = hospitals_id
        self.learning_task = learning_task
        self.learning_rate = learning_rate
        # Metrics initialization
        self.metrics = metrics

        log(INFO, f"Client Name: {self.client_name} Client hospitals {self.hospitals}")

        # Checkpointing: create a string of the names of the hospitals
        self.hospital_names = ",".join(self.hospitals)
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.hospital_names}_best_model.pkl"
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_dict_type(config, "batch_size", int)

        if self.learning_task == "mortality":
            self.model = FendaModel(LocalMLP(), GlobalMLP(), FendaClassifier(FendaJoinMode.CONCATENATE)).to(
                self.device
            )
            # Load training and validation data from the given hospitals.
            self.train_loader, self.val_loader, self.num_examples = load_train_mortality(
                self.data_path, batch_size, self.hospitals
            )
        else:
            self.model = FendaModel(
                LocalMlpD(), GlobalMlpD(), FendaClassifierD(FendaJoinMode.CONCATENATE, size=256)
            ).to(self.device)
            self.train_loader, self.val_loader, self.num_examples = load_train_delirium(
                self.data_path, batch_size, self.hospitals
            )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

        super().setup_client(config)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        meter = AccumulationMeter(self.metrics, "train_meter")

        self.set_parameters(parameters, config)
        local_epochs = self.narrow_dict_type(config, "local_epochs", int)
        current_server_round = self.narrow_dict_type(config, "current_server_round", int)

        metric_values = self.train_by_epochs(current_server_round, local_epochs, meter)
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            metric_values,
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple[float, int, dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        current_server_round = self.narrow_dict_type(config, "current_server_round", int)
        meter = AccumulationMeter(self.metrics, "val_meter")
        loss, metric_values = self.validate(current_server_round, meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def train_by_epochs(
        self,
        current_server_round: int,
        epochs: int,
        meter: Meter,
    ) -> dict[str, Scalar]:
        self.model.train()
        for local_epoch in range(epochs):
            meter.clear()
            for input, target in self.train_loader:
                input, target = input.to(self.device), target.to(self.device)
                # forward pass on the model
                preds = self.model(input)
                train_loss = self.criterion(preds, target)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                meter.update(preds, target)

            log(INFO, f"Local Epoch: {local_epoch}")

        # Return final training metrics
        return meter.compute()

    def validate(self, current_server_round: int, meter: Meter) -> tuple[float, dict[str, Scalar]]:
        self.model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                preds = self.model(input)
                val_loss = self.criterion(preds, target)

                val_loss_sum += val_loss.item()

                meter.update(preds, target)

        metrics = meter.compute()

        val_loss_per_step = val_loss_sum / len(self.val_loader)
        self._maybe_checkpoint(val_loss_per_step)

        return val_loss_per_step, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--hospital_id", nargs="+", default=["THPC", "SMH"], help="ID of hospitals")
    parser.add_argument("--task", action="store", type=str, default="mortality", help="GEMINI usecase: mortality")

    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
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
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=0.001
    )

    args = parser.parse_args()

    if args.task == "mortality":
        data_path = Path("mortality_data")
    elif args.task == "delirium":
        data_path = Path("delirium_data")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Task: {args.task}")
    log(INFO, f"Server Address: {args.server_address}")

    client = GeminiFendaClient(
        data_path,
        [BinaryRocAuc(), BinaryF1(), Accuracy()],
        args.hospital_id,
        device,
        args.task,
        args.learning_rate,
        args.artifact_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
