import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

# from torch.testing import assert_close
from torch.utils.data import DataLoader

from fl4health.clients.flexible.base import FlexibleClient


class _TestFlexibleClient(FlexibleClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        if "global" in self.optimizers:
            return self.optimizers["global"]

        if "local" in self.optimizers:
            return self.optimizers["local"]

        raise RuntimeError("_TestFlexibleClient must have `global` or `local` key set in its `optimizers` attribute.")

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


class _DummyParent:
    def __init__(self) -> None:
        pass
