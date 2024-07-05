import os
import pickle
from logging import INFO
from os.path import exists, join
from typing import Any, Dict, Optional, Tuple

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from flwr.common.logger import log
from flwr.common.typing import Config, Metrics
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
from numpy import ceil
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient, TorchTargetType
from fl4health.utils.metrics import Metric, MetricManager


class nnUNetMetric(Metric):
    def __init__(self, name: str, metric: Metric, has_regions: bool, ignore_label: Optional[int]) -> None:
        """
        Thin wrapper on FL4Health Metric to make it compatible with nnUNet models.
        Requires information from the nnunet trainer (has_regions and ignore_label)

        Args:
            name (str): Name of the metric
            metric (Metric): FL4Health Metric class based metric
            has_regions (bool): Whether or not nnunet region based training
                is being used
            ignore_label (Optional[int]): Used by nnunet to ignore certain
                pixels in images with sparse annotations.
                see (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/ignore_label.md)
        """
        super().__init__(name=name)
        self.metric = metric
        self.ignore_label = ignore_label
        self.has_regions = has_regions

    def maybe_ohe_target(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Checks if targets are one hot encoded and if not one hot encodes them

        Args:
            pred (torch.Tensor): The model predictions
            targets (torch.Tensor): The targets for the predictions

        Returns:
            torch.Tensor: the one hot encoded targets
        """
        # nnUNet default loss require inputs/preds to be one-hot-encoded (ohe)
        # and targets to not be ohe
        # torchmetrics require preds and targets to either both be ohe or both \
        # not be ohe.
        # Therefore check if targets are ohe and if not ohe them

        # Add channel dimension if there isn't one
        if pred.ndim != target.ndim:
            target = target.view(target.shape[0], 1, *target.shape[1:])

        # One hot encode targets if needed
        if pred.shape != target.shape:
            target_ohe = self.ohe(
                ohe_shape=pred.shape, input=target, device=pred.device, dtype=torch.bool
            )  # dtype of ohe targets is boolean

        return target_ohe

    def ohe(
        self, ohe_shape: torch.Size, input: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        One hot encodes (ohe) a torch.Tensor

        Args:
            input (torch.Tensor): the tensor to be one hot encoded
            device (torch.device): the device to put the output ohe tensor on
            dtype (torch.dtype): the desired dtype for the ohe tensor

        Returns:
            torch.Tensor: A ohe version of the input tensor on the specified
                device with the specified dtype
        """
        output_ohe = torch.zeros(ohe_shape, device=device, dtype=dtype)
        # This is how nnunet does ohe in their functions
        # Its a weird function that is not intuitive
        # CAREFUL: Notice the underscore at the end of the scatter function.
        # It makes a difference, was a hard bug to find!
        output_ohe.scatter_(1, input.long(), 1)
        return output_ohe

    def get_oheseg_from_logits(self, input: torch.Tensor) -> torch.Tensor:
        """
        Converts output logits to one hot encoded (ohe) predicted segmentation maps

        Args:
            input (torch.Tensor): the nnunet model ohe output logits with shape
                (batch, classes, x, y(, z))

        Returns:
            torch.Tensor: a one hot encoded predicted segmentation map. All
            values are binary. Has shape (batch, classes, x, y(, z))
        """
        if self.has_regions:
            # If nnunet has_regions, that means pixels can have multiple classes
            pred_ohe = torch.sigmoid(input) > 0.5
        else:
            # produce a non-ohe segmentation map from the ohe logits
            pred_seg = input.argmax(1)[:, None]
            # now ohe the predictions again
            pred_ohe = self.ohe(
                ohe_shape=input.shape, input=pred_seg, device=input.device, dtype=torch.float32
            )  # idk why nnunet sets preds to floats since they should be integers here

        return pred_ohe

    def mask_data(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masks the input and target tensors according to nnunet ignore_label.
        The number of classes in the input tensors should be at least 3
        corresponding to 2 classes for binary segmentation and 1 class which is
        the ignore class specified by ignore label. See nnunet documentation
        for more info

        Args:
            input (torch.Tensor): The one hot encoded predicted
                segmentation maps with shape (batch, classes, x, y(, z))
            target (torch.Tensor): The ground truth segmentation map with shape
                (batch, classes, x, y(, z))

        Returns:
            torch.Tensor: The ohe predicted segmentation maps, maybe masked
            torch.Tensor: The target segmentation maps, maybe masked
        """
        # create mask where 1 is where pixels in target are not ignore label
        # Modify target to remove the last class which is the ignore_label class
        new_target = target
        if self.has_regions:  # target is ohe
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]  # omit the last class
            else:
                mask = 1 - target[:, -1:]
            new_target = new_target[:, :-1]  # Remove final ignore class from target
        else:  # target is not one hot encoded
            mask = (target != self.ignore_label).float()
            # Set ignore label to background essentially removing it as a class
            new_target[new_target == self.ignore_label] = 0

        # Tile the mask to be one hot encoded
        mask_here = torch.tile(mask, (1, input.shape[1], *[1 for _ in range(2, input.ndim)]))

        return input * mask_here, new_target

    def update(self, inputs: TorchTargetType, targets: TorchTargetType) -> None:
        """
        Updates the state of the metric after transforming the preds and
            targets (from the nnunet default format) to be compatible with
            torchmetrics

        Args:
            inputs (TorchTargetType): The output of the nnunet model. Either a
                torch.Tensor or a list of torch.Tensors
            targets (TorchTargetType): The nnunet targets used for the loss
                function
        """
        # When deep supervision is on, the nnunet models have multiple
        # segmentation heads and return lists of torch.Tensors.
        # The nnunet dataloaders correspondingly return lists of torch.Tensors # for the targets

        # Since TorchTargetType is a TypeVar, we know pred and target will be
        # the same type
        if isinstance(inputs, (tuple, list)):
            # If pred is a list or tuple, then assume this is because deep
            # supervision is on. Actual pred and target are at index 0
            input = inputs[0]
            target = targets[0]
        else:
            input = inputs
            target = targets

        # By default, nnunet models output logits. Convert to ohe segmentation maps
        pred_ohe = self.get_oheseg_from_logits(input)

        # Mask data if nnunet ignore label functionality is in use
        if self.ignore_label is not None:
            pred_ohe, target = self.mask_data(pred_ohe, target)

        # one hot encode the target
        target_ohe = self.maybe_ohe_target(input, target)

        # Update metric with ohe predicted and ground truth segmentation maps
        self.metric.update(pred_ohe.long(), target_ohe.long())

    def compute(self, name: Optional[str]) -> Metrics:
        """
        Compute value of underlying TorchMetric.

        Args:
            name (Optional[str]): Optional name used in conjunction with \
                class attribute name to define key in metrics dictionary.

        Returns:
            Metrics: A dictionary of string and Scalar representing the \
                computed metric and its associated key.
        """
        return self.metric.compute(name=name)

    def clear(self) -> None:
        self.metric.clear()


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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
        self.dataset_name = convert_id_to_dataset_name(self.dataset_id)
        self.dataset_json = load_json(join(nnUNet_raw, self.dataset_name, "dataset.json"))
        self.data_identifier = data_identifier
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

    def create_plans(self, config: Config) -> Dict[str, Any]:
        """
        Modifies the provided plans file to work with the local client dataset

        Args:
            config (Config): The config provided by the server. Expects the
                'nnunet_plans' key with a pickled dictionary as the value

        Returns:
            Dict[str, Any]: The modified nnunet plans for the client
        """

        # Get the nnunet plans specified by the server
        plans = pickle.loads(config["nnunet_plans"])  # type: ignore

        # Change plans name
        if self.plans_name is None:
            self.plans_name = f"FL_Dataset{self.dataset_id:03d}" + "-" + plans["plans_name"]
        plans["plans_name"] = self.plans_name

        # Change dataset name
        plans["dataset_name"] = self.dataset_name

        # Change data identifier and ensure batch size is within nnunet limits
        # =====================================================================
        if self.data_identifier is None:
            self.data_identifier = self.plans_name

        # Max batch size for nnunet is 5 percent of dataset
        num_samples = self.dataset_json["numTraining"]
        bs_5percent = round(num_samples * 0.05)

        # Iterate through nnunet configs in plans file
        for c in plans["configurations"].keys():
            # Change the data identifier
            if "data_identifier" in plans["configurations"][c].keys():
                plans["configurations"][c]["data_identifier"] = self.data_identifier + "_" + c
            # Ensure batch size is at least 2, then at most 5 percent of dataset
            if "batch_size" in plans["configurations"][c].keys():
                old_bs = plans["configurations"][c]["batch_size"]
                new_bs = max(min(old_bs, bs_5percent), 2)
                plans["configurations"][c]["batch_size"] = new_bs

        # Can't run nnunet preprocessing without saving plans file
        if not exists(join(nnUNet_preprocessed, self.dataset_name)):
            os.makedirs(join(nnUNet_preprocessed, self.dataset_name))
        plans_save_path = join(nnUNet_preprocessed, self.dataset_name, self.plans_name + ".json")
        save_json(plans, plans_save_path, sort_keys=False)

        return plans

    def maybe_preprocess(self, config: Config) -> None:
        """
        Checks if preprocessed data for current plans exists and if not preprocesses the nnunet_raw dataset

        Args:
            config (Config): The config file provided by the server. Expects
                the 'nnunet_config' key with a string value of either 2d,
                3d_fullres, 3d_lowres, 3d_cascade_fullres
        """
        assert self.data_identifier is not None, 'Was expecting data identifier to be initialized in self.create_plans'
        assert isinstance(config['nnunet_config'], str), 'Was expecting nnunet_config to be a string'
        preprocess_folder = join(
            nnUNet_preprocessed, self.dataset_name, self.data_identifier + "_" + str(config["nnunet_config"])
        )

        # Preprocess data if it's not already there
        if self.always_preprocess or not exists(preprocess_folder):
            default_num_processes = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
            if config["nnunet_config"] in default_num_processes:
                num_processes = default_num_processes[config["nnunet_config"]]
            else:
                num_processes = 4
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_name,
                num_processes=[
                    num_processes
                ],  # Since we only run one config at a time, we know num_processes is an int, change it to a list
                configurations=[config["nnunet_config"]],
            )
        else:
            log(INFO, "nnunet preprocessed data seems to already exist. Skipping preprocessing")

    def wrap_metrics(self) -> None:
        """
        Wraps the provided fl4health.utils.metrics.Metrics with the nnUNetMetric
        class and then updates the clients metric managers
        """
        # Wrap metrics
        new_metrics = []
        for metric in self.metrics:  # self.metrics from parent class
            new_metrics.append(nnUNetMetric(
                name=metric.name,
                metric=metric,
                has_regions=self.nnunet_trainer.label_manager.has_regions,
                ignore_label=self.nnunet_trainer.label_manager.ignore_label,
            ))
        self.metrics = new_metrics

        # Reinstastiate metric managers with wrapped metrics
        self.train_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="train")
        self.val_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="val")
        self.test_metric_manager = MetricManager(metrics=self.metrics, metric_manager_name="test")

    def setup_client(self, config: Config) -> None:
        """
        Ensures the necessary files for training are on disk and initializes
        several class attributes that depend on values in the config from the
        server

        Args:
            config (Config): The config file from the server. The nnUNetClient
                expects the keys 'nnunet_config', 'nnunet_plans', and 'fold' in
                addition to those required by BasicClient
        """
        log(INFO, "Setting up the nnUNetClient")

        # Get the dataset json and dataset name of the local client dataset
        self.plans = self.create_plans(config=config)

        # Create the nnunet trainer
        self.nnunet_trainer = nnUNetTrainer(
            plans=self.plans,
            configuration=config["nnunet_config"],
            fold=config["fold"],
            dataset_json=self.dataset_json,
            device=self.device,
        )

        # I will try and add support for deep supervision later
        self.nnunet_trainer.enable_deep_supervision = False
        self.nnunet_trainer.initialize()

        # Check if dataset fingerprint has been extracted
        if not exists(join(nnUNet_preprocessed, "dataset_fingerprint.json")):
            extract_fingerprints(dataset_ids=[self.dataset_id])

        # Preprocess nnunet_raw data if needed
        self.maybe_preprocess(config=config)

        # Wrap the metrics with nnUNetMetric.
        self.wrap_metrics()

        # Parent function sets up optimizer, criterion, parameter_exchanger, dataloaders and reporters.
        # We have to run this at the end since it depends on the previous setup
        super().setup_client(config)
