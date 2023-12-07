import argparse
from logging import INFO
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from sklearn.preprocessing import MaxAbsScaler
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from examples.models.mlp_classifier import MLP
from fl4health.clients.tabular_data_client import TabularDataClient
from fl4health.utils.metrics import Accuracy, Metric


class Mimic3TabularDataClient(TabularDataClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        id_column: str,
        targets: Union[str, List[str]],
    ) -> None:
        super().__init__(data_path, metrics, device, id_column, targets)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        # random train-valid split.
        indices = np.random.permutation(self.aligned_features.shape[0])
        shuffled_data = self.aligned_features[indices]
        shuffled_targets = self.aligned_targets[indices]
        split_percentage = 0.9
        split_point = int(shuffled_data.shape[0] * split_percentage)
        train_data = shuffled_data[:split_point]
        val_data = shuffled_data[split_point:]
        train_targets = shuffled_targets[:split_point]
        val_targets = shuffled_targets[split_point:]

        tensor_train_data = torch.from_numpy(train_data.toarray()).float()
        tensor_train_targets = torch.from_numpy(train_targets)
        tensor_val_data = torch.from_numpy(val_data.toarray()).float()
        tensor_val_targets = torch.from_numpy(val_targets)

        tensor_train_targets = torch.squeeze(tensor_train_targets.long(), dim=1)
        tensor_val_targets = torch.squeeze(tensor_val_targets.long(), dim=1)

        train_loader = DataLoader(
            TensorDataset(tensor_train_data, tensor_train_targets), batch_size=batch_size, shuffle=True
        )

        val_loader = DataLoader(TensorDataset(tensor_val_data, tensor_val_targets), batch_size=batch_size)
        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        model = MLP(self.input_dimension, self.output_dimension)
        model.to(self.device)
        return model

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.05, weight_decay=0.001)

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_data_frame(self, config: Config) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_path", action="store", type=str, help="Path to the local dataset", default="examples/datasets"
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.dataset_path)

    log(INFO, f"Device to be used: {DEVICE}")
    log(INFO, f"Server Address: {args.server_address}")

    # ham_id is the id column and LOSgroupNum is the target column.
    client = Mimic3TabularDataClient(data_path, [Accuracy("accuracy")], DEVICE, "hadm_id", ["LOSgroupNum"])
    # This call demonstrates how the user may specify a particular sklearn pipeline for a specific feature.
    client.preset_specific_pipeline("NumNotes", MaxAbsScaler())
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
