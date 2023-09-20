import argparse
from logging import INFO
from pathlib import Path
from typing import Dict, Sequence, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArray, Scalar
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.basic_client import BasicClient
from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder
from fl4health.feature_alignment.tab_features_preprocessor import TabularFeaturesPreprocessor
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.metrics import Accuracy, Metric


class TabularDataClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], device: torch.device) -> None:
        super().__init__(data_path, metrics, device)
        self.parameter_exchanger: FullParameterExchanger
        self.tabular_features_info_encoder: TabFeaturesInfoEncoder
        self.tabular_features_preprocessor: TabularFeaturesPreprocessor
        self.df: pd.DataFrame
        self.input_dimension: int
        self.target_dimension: int

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        batch_size = self.narrow_config_type(config, "batch_size", int)
        id_column = self.narrow_config_type(config, "id_column", str)
        target_column = self.narrow_config_type(config, "target_column", str)
        format_specified = self.narrow_config_type(config, "format_specified", bool)

        self.df = pd.read_csv(self.data_path)
        self.tabular_features_info_encoder = TabFeaturesInfoEncoder.encoder_from_dataframe(
            self.df, id_column, target_column
        )

        if format_specified:
            self.tabular_features_info_encoder = TabFeaturesInfoEncoder.from_json(
                self.narrow_config_type(config, "feature_info", str)
            )
            self.tabular_features_preprocessor = TabularFeaturesPreprocessor(self.tabular_features_info_encoder)
            aligned_features, targets = self.tabular_features_preprocessor.preprocess_features(self.df)
            self.input_dimension = aligned_features.shape[1]
            self.target_dimension = targets.shape[1]
            self.train_loader, self.val_loader = self._setup_data_loaders(aligned_features, targets, batch_size)

    def _setup_data_loaders(self, data: NDArray, targets: NDArray, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        split_percentage = 0.9
        split_point = int(len(shuffled_data) * split_percentage)

        train_data = shuffled_data[:split_point]
        val_data = shuffled_data[split_point:]
        train_targets = shuffled_targets[:split_point]
        val_targets = shuffled_targets[split_point:]

        tensor_train_data = torch.tensor(train_data, dtype=torch.float32)
        tensor_train_targets = torch.tensor(train_targets, dtype=torch.float32)
        tensor_val_data = torch.tensor(val_data, dtype=torch.float32)
        tensor_val_targets = torch.tensor(val_targets, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(tensor_train_data, tensor_train_targets), batch_size=batch_size, shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(tensor_val_data, tensor_val_targets), batch_size=batch_size, shuffle=True
        )

        return train_loader, val_loader

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties of client.
        First initializes the client because this is called prior to the first
        federated learning round.
        """
        self.setup_client(config)
        format_specified = self.narrow_config_type(config, "format_specified", bool)
        if not format_specified:
            return {
                "num_train_samples": self.num_examples["train_set"],
                "feature_info": self.tabular_features_info_encoder.to_json(),
            }
        else:
            return {
                "num_train_samples": self.num_examples["train_set"],
                "feature_info": self.tabular_features_info_encoder.to_json(),
                "input_dimension": self.input_dimension,
                "target_dimension": self.target_dimension,
            }


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

    client = TabularDataClient(data_path, [Accuracy("accuracy")], DEVICE)
    # grpc_max_message_length is reset here so the entire model can be exchanged between the server and clients.
    # Note that the server must be started with the same grpc_max_message_length. Otherwise communication
    # of larger messages would still be blocked.
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
