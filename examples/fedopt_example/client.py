import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.fedopt_example.client_data import LabelEncoder, Vocabulary, construct_dataloaders
from examples.fedopt_example.metrics import CustomMetricMeter, MetricMeter
from examples.models.lstm_model import LSTM
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeter, LossMeterType
from fl4health.utils.metrics import MetricMeterManager


class NewsClassifierClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super(BasicClient, self).__init__(data_path, device)
        self.checkpointer = checkpointer
        self.train_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)
        self.val_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)

        self.model: nn.Module
        self.optimizer: torch.optim.Optimizer

        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.num_train_samples: int
        self.num_val_samples: int
        self.learning_rate: float
        self.weight_matrix: torch.Tensor
        self.vocabulary: Vocabulary
        self.label_encoder: LabelEncoder
        self.batch_size: int

        # Need to track total_steps across rounds for WANDB reporting
        self.total_steps: int = 0

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        sequence_length = self.narrow_config_type(config, "sequence_length", int)
        self.batch_size = self.narrow_config_type(config, "batch_size", int)

        train_loader, validation_loader, _, weight_matrix = construct_dataloaders(
            self.data_path, self.vocabulary, self.label_encoder, sequence_length, self.batch_size
        )
        self.weight_matrix = weight_matrix

        return train_loader, validation_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss(weight=self.weight_matrix)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.001)

    def get_model(self, config: Config) -> nn.Module:
        vocab_dimension = self.narrow_config_type(config, "vocab_dimension", int)
        hidden_size = self.narrow_config_type(config, "hidden_size", int)
        return LSTM(self.vocabulary.vocabulary_size, vocab_dimension, hidden_size)

    def setup_client(self, config: Config) -> None:
        self.vocabulary = Vocabulary.from_json(self.narrow_config_type(config, "vocabulary", str))
        self.label_encoder = LabelEncoder.from_json(self.narrow_config_type(config, "label_encoder", str))
        # Define mapping from prediction key to meter to pass to MetricMeterManager constructor for train and val
        train_key_to_meter_map: Dict[str, MetricMeter] = {"prediction": CustomMetricMeter(self.label_encoder)}
        self.train_metric_meter_mngr = MetricMeterManager(train_key_to_meter_map)
        val_key_to_meter_map: Dict[str, MetricMeter] = {"prediction": CustomMetricMeter(self.label_encoder)}
        self.val_metric_meter_mngr = MetricMeterManager(val_key_to_meter_map)
        super().setup_client(config)

    def predict(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # While this isn't optimal, this is a good example of a custom predict function to manipulate the predictions
        assert isinstance(self.model, LSTM)
        h0, c0 = self.model.init_hidden(self.batch_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        preds = self.model(input, (h0, c0))
        return {"prediction": preds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    # Load model and data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = NewsClassifierClient(data_path, DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
