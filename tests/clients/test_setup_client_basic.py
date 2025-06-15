from pathlib import Path

import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from tests.test_utils.models_for_test import LinearModel


class ClientForTest(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        val_loader = DataLoader(TensorDataset(torch.ones((4, 4)), torch.ones((4))))
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return LinearModel().to(self.device)


def test_setup_client() -> None:
    client = ClientForTest(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.setup_client({})
    assert client.parameter_exchanger is not None
    assert client.model is not None
    assert client.optimizers is not None
    assert client.train_loader is not None
    assert client.val_loader is not None
    assert client.num_train_samples is not None
    assert client.num_val_samples is not None
    assert client.initialized
