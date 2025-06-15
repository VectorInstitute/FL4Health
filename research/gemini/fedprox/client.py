import argparse
import os
from logging import INFO
from pathlib import Path

import flwr as fl
import torch
from data.data import load_train_delirium, load_train_mortality
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.metrics import AccumulationMeter, Metric
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc
from research.gemini.mortality_models.NN import NN as mortality_model


class GeminiFedProxClient(FedProxClient):
    def __init__(
        self,
        data_path: Path,
        metrics: list[Metric],
        hospitals_id: list[str],
        device: torch.device,
        learning_task: str,
        learning_rate: float,
        mu: float,
        checkpoint_stub: str,
        run_name: str = "",
    ) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)
        self.hospitals = hospitals_id
        self.learning_task = learning_task
        log(INFO, f"Client Name: {self.client_name} Client hospitals {self.hospitals}")

        # Checkpointing: create a string of the names of the hospitals
        self.hospital_names = ",".join(self.hospitals)
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.hospital_names}_best_model.pkl"
        self.learning_rate = learning_rate
        self.mu = mu
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    def setup_client(self, config: Config) -> None:
        batch_size = self.narrow_dict_type(config, "batch_size", int)

        if self.learning_task == "mortality":
            self.model: nn.Module = mortality_model(input_dim=35, output_dim=1).to(self.device)
            # Load training and validation data from the given hospitals.
            self.train_loader, self.val_loader, self.num_examples = load_train_mortality(
                self.data_path, batch_size, self.hospitals
            )
        else:
            self.model: nn.Module = delirium_model(input_dim=8093, output_dim=1).to(self.device)
            self.train_loader, self.val_loader, self.num_examples = load_train_delirium(
                self.data_path, batch_size, self.hospitals
            )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Set the Proximal Loss weight mu
        self.proximal_weight = self.mu

        self.parameter_exchanger = FullParameterExchanger()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--hospital_id", nargs="+", default=["THPC", "SMH"], help="ID of hospitals")
    parser.add_argument(
        "--task", action="store", type=str, default="mortality", help="GEMINI usecase: mortality, delirium"
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
    parser.add_argument("--mu", action="store", type=float, help="Mu value for the FedProx training", default=0.1)
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=0.01
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
    client = GeminiFedProxClient(
        data_path,
        [BinaryRocAuc(), BinaryF1(), Accuracy()],
        args.hospital_id,
        device,
        args.task,
        args.learning_rate,
        args.mu,
        args.artifact_dir,
        args.run_name,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

    # Shutdown the client gracefully
    client.shutdown()
