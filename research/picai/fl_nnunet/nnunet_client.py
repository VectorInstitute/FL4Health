import os
from os.path import exists, join
from typing import Any, Dict, Literal, Optional, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from numpy import ceil
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient

nnUNetConfig = Literal["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


class nnUNetDLWrapper(DataLoader):
    """Wraps the nnUNetDataLoader class using the pytorch dataloader to avoid typing errors"""

    def __init__(self, nnunet_dataloader: nnUNetDataLoaderBase, infinite: bool = True) -> None:
        """
        Wraps nnunet dataloader classes to make them pytorch compatible

        Args:
            nnunet_dataloader (nnUNetDataLoaderBase): The nnunet dataloader
            infinite (bool, optional): Whether or not to treat the dataset
                as infinite. Defaults to True.
        """
        self.nnunet_dataloader = nnunet_dataloader
        # nnUNetDataloaders store their datasets under the self.data attribute
        self.dataset: nnUNetDataset = nnunet_dataloader.data

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite
        self.len = len(self)

    def __next__(self) -> Tuple[Tuple, Tuple]:
        if not self.infinite and self.current_step == self.len:
            self.reset()
            raise StopIteration
        else:
            self.current_step += 1
            batch = next(self.nnunet_dataloader)  # This returns a dictionary
            return batch["data"], batch["target"]

    def __len__(self) -> int:
        # Should return num_samples // batch_size
        # nnUNetDataloaders are 'infinite' meaning they randomly sample batches \
        # from the dataset. This makes the distinction between \
        # num_samples // batch_size and ceil(num_samples/batch_size) \
        # meaningless. We will abritrarily use the later
        num_samples = len(self.dataset)
        batch_size = self.nnunet_dataloader.batch_size
        return ceil(num_samples / batch_size)

    def reset(self) -> None:
        self.current_step = 0


class Module2LossWrapper(_Loss):
    """Converts a nn.Module subclass to a _Loss subclass"""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super.__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss(pred, target)


class nnUNetClient(BasicClient):
    def __init__(
        self,
        dataset_id: int,
        data_identifier: Optional[str],
        save_plans: Optional[str | bool],
        always_preprocess: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_id: int = dataset_id
        self.data_identifier: Optional[str] = data_identifier  # Do not include config
        self.device = kwargs["device"]
        self.always_preprocess: bool = always_preprocess
        self.save_plans: Optional[str | bool] = save_plans

        if type(self.save_plans) is str:
            self.plans_name: Optional[str] = self.save_plans
        else:
            self.plans_name = None

    def get_data_loaders(self, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        # Get the nnunet dataloader iterators
        train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Wrap the iterators in another class to make them pytorch iterators
        # And avoid typing errors
        train_loader = nnUNetDLWrapper(train_loader)
        val_loader = nnUNetDLWrapper(val_loader)

        return train_loader, val_loader

    def get_model(self, config: Dict[str, Any]) -> nn.Module:
        network = self.nnunet_trainer.build_network_architecture(
            architecture_class_name=self.nnunet_trainer.configuration_manager.network_arch_class_name,
            arch_init_kwargs=self.nnunet_trainer.configuration_manager.network_arch_init_kwargs,
            arch_init_kwargs_req_import=self.nnunet_trainer.configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels=self.nnunet_trainer.num_input_channels,
            num_output_channels=self.nnunet_trainer.label_manager.num_segmentation_heads,
            enable_deep_supervision=self.nnunet_trainer.enable_deep_supervision,
        ).to(self.nnunet_trainer.device)

        return network

    def get_criterion(self, config: Dict[str, Any]) -> _Loss:
        return Module2LossWrapper(self.nnunet_trainer._build_loss())

    def get_optimizer(self, config: Dict[str, Any]) -> Optimizer:
        optimizer, lr_scheduler = self.nnunet_trainer.configure_optimizers()
        return optimizer

    def setup_client(self, config: Dict[str, Any]) -> None:
        print("Setting Up Client")
        # Get the nnunet plans specified by the server
        print(config)
        plans = config["nnunet_plans"]

        # Get the dataset json of the local client dataset
        dataset_name = convert_id_to_dataset_name(self.dataset_id)
        dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))

        # Change plans name
        if self.plans_name is None:
            self.plans_name = f"FL_Dataset{self.dataset_id:03d}" + plans["plans_name"]
        plans["plans_name"] = self.plans_name

        # Change dataset name
        plans["dataset_name"] = dataset_name

        # Change data identifier and ensure batch size is within limits
        if self.data_identifier is None:
            self.data_identifier = self.plans_name
        num_samples = dataset_json["numTraining"]
        bs_5percent = round(num_samples * 0.05)  # Set max batch size to 5 percent of dataset
        for c in plans["configurations"].keys():
            if "data_identifier" in plans["configurations"][c].keys():
                plans["configurations"][c]["data_identifier"] = self.data_identifier + "_" + c

            if "batch_size" in plans["configurations"][c].keys():
                old_bs = plans["configurations"][c]["batch_size"]
                new_bs = max(min(old_bs, bs_5percent), 2)  # Min 2, max 5 percent of dataset
                plans["configurations"][c]["batch_size"] = new_bs

        if self.save_plans:
            if not exists(join(nnUNet_preprocessed, dataset_name)):
                os.makedirs(join(nnUNet_preprocessed, dataset_name))
            plans_save_path = join(nnUNet_preprocessed, dataset_name, self.plans_name + ".json")
            save_json(plans, plans_save_path, sort_keys=False)

        # Create the nnunet trainer
        self.nnunet_trainer = nnUNetTrainer(
            plans=plans,
            configuration=config["nnunet_config"],
            fold=config["fold"],
            dataset_json=dataset_json,
            device=self.device,
        )

        # Parent function sets up optimizer, criterion, parameter_exchanger, dataloaders and reporters.
        # We have to run this after creating the nnUNetTrainer since it calls functions that depend on that attribute
        # Need to change config file to only have ints as values
        super_config = config.copy()
        for key, value in super_config.items():
            if type(value) is not int:
                del super_config[key]
        super().setup_client(super_config)

        # Check if dataset fingerprint has been extracted
        if not exists(join(nnUNet_preprocessed, "dataset_fingerprint.json")):
            extract_fingerprints(dataset_ids=[self.dataset_id])

        # Check if the preprocessed data already exists
        preprocess_folder = join(
            nnUNet_preprocessed, dataset_name, self.data_identifier + str(config["nnunet_config"])
        )
        if self.always_preprocess or not exists(preprocess_folder):
            default_num_processes = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
            if config["nnunet_config"] in default_num_processes:
                num_processes = default_num_processes[config["nnunet_config"]]
            else:
                num_processes = 4
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_name,
                num_processes=num_processes,
                configurations=[config["nnunet_config"]],
            )
