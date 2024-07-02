import os
import pickle
from os.path import exists, join
from typing import Any, Literal, Optional, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from flwr.common.typing import Config
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

# class nnUNetMetricManager(MetricManager):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def update(self, preds: Dict[str, Tensor], target: Tensor) -> None:
#         # nnUNet models output seperate channels for each class (even in binary segmentation)
#         # However labels are not onehot encoded to match leading to a dimension error when using torchmetrics
#         # We only have a single target
#         pred_key: str = next(iter(preds))
#         pred: Tensor = preds[pred_key]
#         if pred.ndim != target.ndim: # Add channel dimension if there isn't one
#             target = target.view(target.shape[0], 1, *target.shape[1:])

#         if pred.shape != target.shape:
#             target_one_hot = torch.zeros(pred.shape, dtype=torch.bool)
# This does the onehot encoding. Its a weird function that is not intuitive
#             target_one_hot.scatter(1, target.long(), 1)

#         return super().update(preds, target_one_hot)


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
        self.dataset: nnUNetDataset = self.nnunet_dataloader.generator._data
        super().__init__(dataset=self.dataset, batch_size=self.nnunet_dataloader.generator.batch_size)

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite
        self.len = len(self.dataset)

    def __next__(self) -> Tuple[Tuple, Tuple]:
        if not self.infinite and self.current_step == self.len:
            self.reset()
            raise StopIteration
        else:
            self.current_step += 1
            batch = next(self.nnunet_dataloader)  # This returns a dictionary
            # Note that this only works right now because I have turned deep supervision off
            # When deep supervision is on, target is a list of segmentations at various scales
            # nnUNet has a wrapper for loss functions to enable deep supervision but it expects
            # target to be a list of tensors instead of a tensor itself
            # This causes an error in basic client which expects target to always be a tensor
            return batch["data"], batch["target"]

    def __len__(self) -> int:
        # Should return num_samples // batch_size
        # nnUNetDataloaders are 'infinite' meaning they randomly sample batches \
        # from the dataset. This makes the distinction between \
        # num_samples // batch_size and ceil(num_samples/batch_size) \
        # meaningless. We will abritrarily use the later
        num_samples = len(self.dataset)
        batch_size = self.nnunet_dataloader.generator.batch_size
        return int(ceil(num_samples / batch_size))

    def reset(self) -> None:
        self.current_step = 0

    def __iter__(self) -> DataLoader:  # type: ignore
        # mypy gets angry that the return type
        return self


class Module2LossWrapper(_Loss):
    """Converts a nn.Module subclass to a _Loss subclass"""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.loss(pred, target)


class nnUNetClient(BasicClient):
    def __init__(
        self,
        dataset_id: int,
        data_identifier: Optional[str],
        plans_identifier: Optional[str],
        always_preprocess: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_id: int = dataset_id
        self.data_identifier: Optional[str] = data_identifier  # Do not include config
        self.device = kwargs["device"]
        self.always_preprocess: bool = always_preprocess
        self.plans_name = plans_identifier

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        # Get the nnunet dataloader iterators
        train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Wrap the iterators in another class to make them pytorch iterators
        # And avoid typing errors
        # By default nnunet dataloaders are infinite, so if we are training by
        # num epochs we need to make them not infinite anymore
        if config["local_epochs"] is not None:
            infinite = False
        train_loader = nnUNetDLWrapper(train_loader, infinite=infinite)
        val_loader = nnUNetDLWrapper(val_loader, infinite=infinite)

        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return self.nnunet_trainer.network

    def get_criterion(self, config: Config) -> _Loss:
        return Module2LossWrapper(self.nnunet_trainer.loss)

    def get_optimizer(self, config: Config) -> Optimizer:
        return self.nnunet_trainer.optimizer

    def setup_client(self, config: Config) -> None:
        print("Setting Up Client")
        # Get the nnunet plans specified by the server
        plans = pickle.loads(config["nnunet_plans"])  # type: ignore

        # Get the dataset json of the local client dataset
        dataset_name = convert_id_to_dataset_name(self.dataset_id)
        dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))

        # Change plans name
        if self.plans_name is None:
            self.plans_name = f"FL_Dataset{self.dataset_id:03d}" + "-" + plans["plans_name"]
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

        # The way it is right now, we have to save the plans file in order to do the preprocessing
        # Only way to avoid this would be to rewrite the preprocessing function which I don't want
        # to do right now
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
        # I will try and add support for deep supervision later
        self.nnunet_trainer.enable_deep_supervision = False
        self.nnunet_trainer.initialize()

        # Check if dataset fingerprint has been extracted
        if not exists(join(nnUNet_preprocessed, "dataset_fingerprint.json")):
            extract_fingerprints(dataset_ids=[self.dataset_id])

        # Check if the preprocessed data already exists
        preprocess_folder = join(
            nnUNet_preprocessed, dataset_name, self.data_identifier + "_" + str(config["nnunet_config"])
        )
        if self.always_preprocess or not exists(preprocess_folder):
            default_num_processes = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
            if config["nnunet_config"] in default_num_processes:
                num_processes = default_num_processes[str(config["nnunet_config"])]
            else:
                num_processes = 4
            print("Preprocessing")
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_name,
                num_processes=[
                    num_processes
                ],  # Since we only run one config at a time, we know num_processes is an int, change it to a list
                configurations=[config["nnunet_config"]],
            )
        else:
            print("Preprocessed data already exists")

        # Parent function sets up optimizer, criterion, parameter_exchanger, dataloaders and reporters.
        # We have to run this at the end since it depends on the previous setup
        super().setup_client(config)
