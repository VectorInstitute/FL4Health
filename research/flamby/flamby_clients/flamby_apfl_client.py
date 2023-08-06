import os
from logging import INFO
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.clients.apfl_client import ApflClient
from fl4health.utils.metrics import AccumulationMeter, Metric


class FlambyApflClient(ApflClient):
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
        self.client_number = client_number
        log(INFO, f"Client Name: {self.client_name}, Client Number: {self.client_number}")
        checkpoint_dir = os.path.join(checkpoint_stub, run_name)
        checkpoint_name = f"client_{self.client_number}_best_model.pkl"
        self.learning_rate = learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)
        self.dataset_dir = dataset_dir

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
