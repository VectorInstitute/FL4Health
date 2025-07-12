import argparse
import os
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from data.data import load_train_delirium, load_train_mortality
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.clients.apfl_client import ApflClient
from fl4health.metrics import AccumulationMeter, Meter, Metric
from fl4health.model_bases.apfl_base import APFLModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc
from research.gemini.mortality_models.NN import NN as mortality_model


LocalLoss = torch.Tensor
GlobalLoss = torch.Tensor
PersonalLoss = torch.Tensor

LocalPreds = torch.Tensor
GlobalPreds = torch.Tensor
PersonalPreds = torch.Tensor

ApflTrainStepOutputs = tuple[LocalLoss, GlobalLoss, PersonalLoss, LocalPreds, GlobalPreds, PersonalPreds]


class GeminiApflClient(ApflClient):
    def __init__(
        self,
        data_path: Path,
        metrics: list[Metric],
        hospitals_id: list[str],
        device: torch.device,
        learning_task: str,
        learning_rate: float,
        alpha_learning_rate: float,
        checkpoint_stub: str,
        run_name: str = "",
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)
        self.hospitals = hospitals_id
        self.learning_task = learning_task
        self.learning_rate = learning_rate
        self.alpha_learning_rate = alpha_learning_rate
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
            self.model: APFLModule = APFLModule(
                mortality_model(input_dim=35, output_dim=1), alpha_lr=self.alpha_learning_rate
            ).to(self.device)
            # Load training and validation data --> from the given hospitals.
            self.train_loader, self.val_loader, self.num_examples = load_train_mortality(
                self.data_path, batch_size, self.hospitals
            )
        else:
            self.model: APFLModule = APFLModule(
                delirium_model(input_dim=8093, output_dim=1), alpha_lr=self.alpha_learning_rate
            ).to(self.device)
            self.train_loader, self.val_loader, self.num_examples = load_train_delirium(
                self.data_path, batch_size, self.hospitals
            )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.global_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.parameter_exchanger = FixedLayerExchanger(self.model.layers_to_exchange())

        super().setup_client(config)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        global_meter = AccumulationMeter(self.metrics, "train_global")
        local_meter = AccumulationMeter(self.metrics, "train_local")
        personal_meter = AccumulationMeter(self.metrics, "train_personal")

        self.set_parameters(parameters, config)
        local_epochs = self.narrow_dict_type(config, "local_epochs", int)

        metric_values = self.train_by_epochs(local_epochs, global_meter, local_meter, personal_meter)
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

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> ApflTrainStepOutputs:
        # Mechanics of training loop follow from original implementation
        # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
        input, target = input.to(self.device), target.to(self.device)

        # Forward pass on global model and update global parameters
        self.global_optimizer.zero_grad()
        global_pred = self.model(input, personal=False)["global"]
        global_loss = self.criterion(global_pred, target)
        global_loss.backward()
        self.global_optimizer.step()

        # Make sure gradients are zero prior to forward passes of global and local model
        # to generate personalized predictions
        # NOTE: We zero the global optimizer grads because they are used (after the backward calculation below)
        # to update the scalar alpha (see update_alpha() where .grad is called.)
        self.global_optimizer.zero_grad()
        self.local_optimizer.zero_grad()

        # Personal predictions are generated as a convex combination of the output
        # of local and global models
        pred_dict = self.model(input, personal=True)
        personal_pred, local_pred = pred_dict["personal"], pred_dict["local"]

        # Parameters of local model are updated to minimize loss of personalized model
        personal_loss = self.criterion(personal_pred, target)
        personal_loss.backward()
        self.local_optimizer.step()

        with torch.no_grad():
            local_loss = self.criterion(local_pred, target)

        return local_loss, global_loss, personal_loss, local_pred, global_pred, personal_pred

    def train_by_epochs(
        self, epochs: int, global_meter: Meter, local_meter: Meter, personal_meter: Meter
    ) -> dict[str, Scalar]:
        self.model.train()
        for epoch in range(epochs):
            loss_dict = {"personal": 0.0, "local": 0.0, "global": 0.0}
            global_meter.clear()
            local_meter.clear()
            personal_meter.clear()

            for step, (input, target) in enumerate(self.train_loader):
                # Mechanics of training loop follow from original implementation
                # https://github.com/MLOPTPSU/FedTorch/blob/main/fedtorch/comms/trainings/federated/apfl.py
                local_loss, global_loss, personal_loss, local_preds, global_preds, personal_preds = self.train_step(
                    input, target
                )

                # Only update alpha if it is the first epoch and first step of training
                # and adaptive alpha is true
                if self.is_start_of_local_training(epoch, step) and self.model.adaptive_alpha:
                    self.model.update_alpha()

                loss_dict["local"] += local_loss.item()
                loss_dict["global"] += global_loss.item()
                loss_dict["personal"] += personal_loss.item()

                global_meter.update(global_preds, target)
                local_meter.update(local_preds, target)
                personal_meter.update(personal_preds, target)

            loss_dict = {key: val / len(self.train_loader) for key, val in loss_dict.items()}

        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        log(INFO, f"Performed {epochs} Epochs of Local training")

        return metrics

    def validate(
        self, global_meter: Meter, local_meter: Meter, personal_meter: Meter
    ) -> tuple[float, dict[str, Scalar]]:
        self.model.eval()
        loss_dict = {"global": 0.0, "personal": 0.0, "local": 0.0}
        global_meter.clear()
        local_meter.clear()
        personal_meter.clear()

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)

                global_pred = self.model(input, personal=False)["global"]
                global_loss = self.criterion(global_pred, target)

                pred_dict = self.model(input, personal=True)
                personal_pred, local_pred = pred_dict["personal"], pred_dict["local"]
                personal_loss = self.criterion(personal_pred, target)
                local_loss = self.criterion(local_pred, target)

                loss_dict["global"] += global_loss.item()
                loss_dict["personal"] += personal_loss.item()
                loss_dict["local"] += local_loss.item()

                global_meter.update(global_pred, target)
                local_meter.update(local_pred, target)
                personal_meter.update(personal_pred, target)

        loss_dict = {key: val / len(self.val_loader) for key, val in loss_dict.items()}
        global_metrics = global_meter.compute()
        local_metrics = local_meter.compute()
        personal_metrics = personal_meter.compute()
        metrics: dict[str, Scalar] = {**global_metrics, **local_metrics, **personal_metrics}
        self._maybe_checkpoint(loss_dict["personal"])
        return loss_dict["personal"], metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--hospital_id", nargs="+", default=["THPC", "SMH"], help="ID of hospitals")
    parser.add_argument(
        "--task", action="store", type=str, default="mortality", help="GEMINI usecase: mortality or delirium"
    )

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
    parser.add_argument(
        "--alpha_learning_rate", action="store", type=float, help="Learning rate for the APFL alpha", default=0.01
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

    client = GeminiApflClient(
        data_path,
        [BinaryRocAuc(), BinaryF1(), Accuracy()],
        args.hospital_id,
        device,
        args.task,
        args.learning_rate,
        args.alpha_learning_rate,
        args.artifact_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
