import pytest
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.flexible.base import FlexibleClient
from fl4health.mixins.personalized import DittoPersonalizedMixin, PersonalizedMode, make_it_personal


class MyClient(FlexibleClient):
    def get_model(self, config: Config) -> nn.Module:
        return self.model

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.val_loader

    def get_optimizer(self, config: Config) -> Optimizer | dict[str, Optimizer]:
        return self.optimizers["global"]

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()


def test_make_it_personal_factory_method() -> None:
    ditto_my_client_cls = make_it_personal(MyClient, mode=PersonalizedMode.DITTO)

    assert issubclass(ditto_my_client_cls, (FlexibleClient, DittoPersonalizedMixin))


def test_make_it_personal_raises_value_error() -> None:
    """Ignore mypy error since a user of the library may not be using static checks in their application."""
    with pytest.raises(ValueError, match="Unrecognized personalized mode."):
        _ = make_it_personal(MyClient, mode="invalid-mode")  # type: ignore [arg-type]
