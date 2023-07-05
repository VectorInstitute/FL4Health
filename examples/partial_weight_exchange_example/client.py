import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from custom_dataloaders import setup_datasets
from flwr.common.typing import Config, NDArrays, Scalar
from torch.utils.data import DataLoader, Dataset
from trainer import infer, train
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.layer_exchanger import NormDriftLayerExchanger


class TransformerPartialExchangeClient(NumpyFlClient):
    def __init__(self, data_path: Path, device: torch.device, model: nn.Module, datasets: Dict[str, Dataset]) -> None:
        super().__init__(data_path, device)
        self.model = model.to(self.device)
        self.initial_model = copy.deepcopy(self.model).to(device)
        self.parameter_exchanger = NormDriftLayerExchanger(1)
        self.datasets = datasets

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)
        self.set_parameters(parameters, config)

        local_epochs = self.narrow_config_type(config, "local_epochs", int)
        accuracy = train(
            self.model,
            self.train_loader,
            self.validation_loader,
            nn.CrossEntropyLoss(),
            n_epochs=local_epochs,
            device=self.device,
        )
        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_examples["train_set"],
            {"accuracy": accuracy},
        )

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        accuracy, loss = infer(self.model, nn.CrossEntropyLoss(), self.validation_loader, device=self.device)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["validation_set"],
            {"accuracy": accuracy},
        )

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)

        self.train_loader = DataLoader(self.datasets["train"], batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.datasets["val"], batch_size=batch_size, shuffle=True)

        self.num_examples = {
            "train_set": len(self.train_loader) * batch_size,
            "validation_set": len(self.validation_loader) * batch_size,
        }

    def get_parameters(self, config: Config) -> NDArrays:
        # Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        # determine parameters sent
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, self.initial_model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        # parameters are set
        assert self.model is not None and self.parameter_exchanger is not None
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        self.parameter_exchanger.pull_parameters(parameters, self.initial_model, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset", default=".")
    args = parser.parse_args()

    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta_classifier_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

    train_datasets, val_datasets, test_datasets = setup_datasets(
        train_split_ratio=0.9, tokenizer=roberta_tokenizer, dataset_name="ag_news", num_clients=2
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    mp.set_start_method("spawn", force=True)
    processes = []

    for train_set, val_set, test_set in zip(train_datasets, val_datasets, test_datasets):
        datasets = {"train": train_set, "val": val_set, "test": test_set}
        client = TransformerPartialExchangeClient(data_path, DEVICE, roberta_classifier_model, datasets)
        p = mp.Process(
            target=fl.client.start_numpy_client, kwargs={"server_address": "0.0.0.0:8080", "client": client}
        )
        p.start()
        processes.append(p)
        print("Process started")

    for p in processes:
        p.join()
