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
from fl4health.clients.scaffold_client import ScaffoldClient
from fl4health.metrics import AccumulationMeter, Meter, Metric
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithControlVariates
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc
from research.gemini.mortality_models.NN import NN as mortality_model


class GeminiScaffoldclient(ScaffoldClient):
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
        super().__init__(data_path=data_path, metrics=metrics, device=device)
        self.hospitals = hospitals_id
        self.learning_task = learning_task
        self.learning_rate_local = learning_rate
        # Metrics initialization
        self.metrics = metrics

        log(INFO, f"Client Name: {self.client_name} Client hospitals {self.hospitals}")

        # Checkpointing: create a string of the names of the hospitals
        self.hospital_names = ",".join(self.hospitals)
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.hospital_names}_best_model.pkl"
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

        # only in Scaffold
        self.batch_size = 64

    def setup_client(self, config: Config) -> None:
        if self.learning_task == "mortality":
            self.model: nn.Module = mortality_model(input_dim=35, output_dim=1).to(self.device)
            # Load training and validation data from the given hospitals.
            self.train_loader, self.val_loader, self.num_examples = load_train_mortality(
                self.data_path, self.batch_size, self.hospitals
            )
        else:
            self.model: nn.Module = delirium_model(input_dim=8093, output_dim=1).to(self.device)
            self.train_loader, self.val_loader, self.num_examples = load_train_delirium(
                self.data_path, self.batch_size, self.hospitals
            )

        self.criterion = torch.nn.BCEWithLogitsLoss()
        # Note that, unlike the other approaches, SCAFFOLD requires a vanilla SGD optimizer for the corrections to
        # make sense mathematically.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate_local)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.self.learning_rate_local, momentum=0.9)

        self.parameter_exchanger = ParameterExchangerWithControlVariates()

        super().setup_client(config)

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        meter = AccumulationMeter(self.metrics, "train_meter")

        self.set_parameters(parameters, config)
        local_steps = self.narrow_dict_type(config, "local_steps", int)

        metric_values = self.train_by_rounds(local_steps, meter)
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
        # current_server_round = self.narrow_dict_type(config, "current_server_round", int)
        meter = AccumulationMeter(self.metrics, "val_meter")
        loss, metric_values = self.validate(meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            metric_values,
        )

    def update_control_variates(self, local_steps: int) -> None:
        """
        Updates local control variates along with the corresponding updates
        according to the option 2 in Equation 4 in https://arxiv.org/pdf/1910.06378.pdf
        To be called after weights of local model have been updated.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None
        assert self.server_model_weights is not None
        assert self.learning_rate_local is not None
        assert self.train_loader.batch_size is not None

        # y_i
        client_model_weights = [val.cpu().numpy() for val in self.model.state_dict().values()]

        # (x - y_i)
        delta_model_weights = self.compute_parameters_delta(self.server_model_weights, client_model_weights)

        # (c_i - c)
        delta_control_variates = self.compute_parameters_delta(
            self.client_control_variates, self.server_control_variates
        )

        updated_client_control_variates = self.compute_updated_control_variates(
            local_steps, delta_model_weights, delta_control_variates
        )
        self.client_control_variates_updates = self.compute_parameters_delta(
            updated_client_control_variates, self.client_control_variates
        )

        # c_i = c_i^plus
        self.client_control_variates = updated_client_control_variates

    def modify_grad(self) -> None:
        """
        Modifies the gradient of the local model to correct for client drift.
        To be called after the gradients have been computed on a batch of data.
        Updates not applied to params until step is called on optimizer.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None

        for param, client_cv, server_cv in zip(
            self.model.parameters(), self.client_control_variates, self.server_control_variates
        ):
            assert param.grad is not None
            tensor_type = param.grad.dtype
            update = torch.from_numpy(server_cv).type(tensor_type) - torch.from_numpy(client_cv).type(tensor_type)
            #         Changed
            param.grad += update.to(self.device)

    def train_by_rounds(
        self,
        local_steps: int,
        meter: Meter,
    ) -> dict[str, Scalar]:
        self.model.train()
        running_loss = 0.0
        meter.clear()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)
        for _ in range(local_steps):
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            input, target = input.to(self.device), target.to(self.device)

            # Forward pass on global model and update global parameters
            self.optimizer.zero_grad()
            pred = self.model(input)
            loss = self.criterion(pred, target)
            loss.backward()

            # modify grad to correct for client drift
            self.modify_grad()
            self.optimizer.step()

            running_loss += loss.item()
            meter.update(pred, target)

        running_loss = running_loss / local_steps

        metrics = meter.compute()
        self.update_control_variates(local_steps)

        return metrics

    def validate(self, meter: Meter) -> tuple[float, dict[str, Scalar]]:
        self.model.eval()
        running_loss = 0.0
        meter.clear()
        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                pred = self.model(input)
                loss = self.criterion(pred, target)

                running_loss += loss.item()
                meter.update(pred, target)

        running_loss = running_loss / len(self.val_loader)
        metrics = meter.compute()

        self._maybe_checkpoint(running_loss)
        return running_loss, metrics


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

    client = GeminiScaffoldclient(
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
