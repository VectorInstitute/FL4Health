from typing import Any, Literal, Tuple

from flwr.common.typing import Config
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient

nnUNetConfig = Literal["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


class nnUNetDLWrapper(DataLoader):
    """Wraps the nnUNetDataLoader class using the pytorch dataloader to avoid typing errors"""

    def __init__(self, nnunet_dataloader: nnUNetDataLoaderBase) -> None:
        """Args:
        nnunet_dataloader: The nnunet data loader iterator
        """
        self.nnunet_dataloader = nnunet_dataloader

    def __next__(self) -> Tuple[Tuple, Tuple]:
        batch = next(self.nnunet_dataloader)
        return batch["data"], batch["target"]  # This returns a dictrionary


class Module2LossWrapper(_Loss):
    """Converts a nn.Module subclass to a _Loss subclass"""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super.__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss(pred, target)


class nnUNetClient(BasicClient):
    def __init__(self, nnunet_plans: dict, config: nnUNetConfig, dataset_json: dict, **kwargs: Any) -> None:
        super.__init__(**kwargs)
        self.nnunet_trainer = nnUNetTrainer(
            plans=nnunet_plans,
            configuration=config,
            fold=0,  # Fold of trainer does not matter for extracting info
            dataset_json=dataset_json,
            device=kwargs["device"],
        )

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        # Get the nnunet dataloader iterators
        train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Wrap the iterators in another class to make them pytorch iterators
        # And avoid typing errors
        train_loader = nnUNetDLWrapper(train_loader)
        val_loader = nnUNetDLWrapper(val_loader)

        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        network = self.nnunet_trainer.build_network_architecture(
            architecture_class_name=self.nnunet_trainer.configuration_manager.network_arch_class_name,
            arch_init_kwargs=self.nnunet_trainer.configuration_manager.network_arch_init_kwargs,
            arch_init_kwargs_req_import=self.nnunet_trainer.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=self.nnunet_trainer.num_input_channels,
            num_output_channels=self.nnunet_trainer.label_manager.num_segmentation_heads,
            enable_deep_supervision=self.nnunet_trainer.enable_deep_supervision,
        ).to(self.nnunet_trainer.device)

        return network

    def get_criterion(self, config: Config) -> _Loss:
        return Module2LossWrapper(self.nnunet_trainer._build_loss())

    def get_optimizer(self, config: Config) -> Optimizer:
        optimizer, lr_scheduler = self.nnunet_trainer.configure_optimizers()
        return optimizer
