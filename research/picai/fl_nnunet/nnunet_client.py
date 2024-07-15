import contextlib
import logging
import os
import pickle
import signal
import sys
import warnings
from logging import INFO
from os.path import exists, join
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, TypeGuard, get_args

import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from numpy import ceil
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.metrics import MetricManager
from fl4health.utils.typing import TorchInputType, TorchPredType, TorchTargetType

with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
    from nnunetv2.training.dataloading.utils import unpack_dataset
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

nnUNetConfig = Literal["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]
# Get the default signal handlers used by python before flwr ovverrides them
og_sigint_handler = signal.getsignal(signal.SIGINT)
og_sigterm_handler = signal.getsignal(signal.SIGTERM)


def exit_gracefully(*args: Any) -> None:
    """A signal handler that does nothing"""
    pass


class DummyFile(object):
    def write(self, x: Any) -> None:
        pass


@contextlib.contextmanager
def nostdout() -> Generator[Any, Any, Any]:
    save_stdout = sys.stdout
    sys.stdout = DummyFile()  # type: ignore
    yield
    sys.stdout = save_stdout


def get_num_spatial_dims(nnunet_config: nnUNetConfig) -> int:
    """
    Returns the number of spatial dimensions given the nnunet configuration

    Args:
        nnunet_config (nnUNetConfig): string specifying the nnunet config

    Returns:
        int: The number of spatial dimensions
    """
    if nnunet_config == "2d":
        return 2
    elif nnunet_config in ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
        return 3
    else:
        raise TypeError(f"Got unexpected nnunet config: {nnunet_config}")


def convert_ds_list_to_dict(
    tensor_list: List[torch.Tensor] | Tuple[torch.Tensor], num_spatial_dims: int
) -> Dict[str, torch.Tensor]:
    """
    Converts a list of torch.Tensors to a dictionary. Names the keys for
    each tensor based on the spatial resolution of the tensor and it's
    index in the list. Useful for nnUNet models with deep supervision where
    model outputs and targets loaded by the dataloader are lists. Assumes the
    tensors have a batch dimension.

    Args:
        tensor_list (List[torch.Tensor]): A list of tensors, usually either
            nnunet model outputs or targets, to be converted into a dictionary
        num_spatial_dims (int): The number of spatial dimensions. Assumes the
            spatial dimensions are last
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the the tensors as
            values where the keys are 'i-XxYxZ' where i was the tensors index
            in the list and X,Y,Z are the spatial dimensions of the tensor
    """
    # Convert list of targets into a dictionary
    tensors = {}

    for i, tensor in enumerate(tensor_list):
        # generate a key based on the spatial dimension and index
        key = str(i) + "-"
        for i in range(num_spatial_dims, 0, -1):
            key += str(tensor.shape[-i]) + "x"
        key = key[:-1]  # remove the trailing 'x'

        tensors.update({key: tensor})

    return tensors


def convert_ds_dict_to_list(tensor_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a dictionary of tensors back into a list so that it can be used
    by nnunet deep supervision loss functions

    Args:
        tensor_dict (Dict[str, torch.Tensor]): Dictionary containing
            torch.Tensors. The key values must start with 'X-' where X is an
            integer representing the index at which the tensor should be placed
            in the output list

    Returns:
        List[torch.Tensor]: A list of torch.Tensors
    """
    tensor_list = []
    for i in range(len(tensor_dict.keys())):  # keys start with index
        key = [k for k in tensor_dict.keys() if k.startswith(str(i))]
        assert (
            len(key) == 1
        ), f"found more than one key that started with {i}. \
            Was expecting the first character to be the index in the \
            list from which the predictions were derived. You're \
            using an nnunet model with deep supervision right?"

        tensor_list.append(tensor_dict[key[0]])

    return tensor_list


def is_valid_config(val: str) -> TypeGuard[nnUNetConfig]:
    return val in list(get_args(nnUNetConfig))


class nnUNetDLWrapper(DataLoader):
    """Wraps the nnUNetDataLoader class using the pytorch dataloader to avoid typing errors"""

    def __init__(
        self, nnunet_dataloader: nnUNetDataLoaderBase, nnunet_config: nnUNetConfig, infinite: bool = True
    ) -> None:
        """
        Wraps nnunet dataloader classes to make them pytorch compatible

        Args:
            nnunet_dataloader (nnUNetDataLoaderBase): The nnunet dataloader
            infinite (bool, optional): Whether or not to treat the dataset
                as infinite. Defaults to True.
        """
        self.nnunet_dataloader = nnunet_dataloader

        # Figure out if dataloader is 2d or 3d
        self.num_spatial_dims = get_num_spatial_dims(nnunet_config)

        # nnUNetDataloaders store their datasets under the self.data attribute
        self.dataset: nnUNetDataset = self.nnunet_dataloader.generator._data
        super().__init__(dataset=self.dataset, batch_size=self.nnunet_dataloader.generator.batch_size)

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite
        self.len = len(self.dataset)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor | Dict[str, torch.Tensor]]:
        if not self.infinite and self.current_step == self.len:
            self.reset()
            raise StopIteration
        else:
            self.current_step += 1
            batch = next(self.nnunet_dataloader)  # This returns a dictionary
            # Note: When deep supervision is on, target is a list of segmentations at various scales
            # nnUNet has a wrapper for loss functions to enable deep supervision
            inputs: torch.Tensor = batch["data"]
            if isinstance(batch["target"], list):
                target_dict = convert_ds_list_to_dict(batch["target"], self.num_spatial_dims)
                return inputs, target_dict
            elif isinstance(batch["target"], torch.Tensor):
                target_tensor: torch.Tensor = batch["target"]
                return inputs, target_tensor
            else:
                raise TypeError(
                    "Was expecting the target generated by the nnunet dataloader to be a list or a torch.Tensor"
                )

    def __len__(self) -> int:
        # Should return num_samples // batch_size
        # nnUNetDataloaders are 'infinite' meaning they randomly sample batches
        # from the dataset. This makes the distinction between
        # num_samples // batch_size and ceil(num_samples/batch_size)
        # meaningless. We will abritrarily use the later
        num_samples = len(self.dataset)
        batch_size = self.nnunet_dataloader.generator.batch_size
        return int(ceil(num_samples / batch_size))

    def reset(self) -> None:
        self.current_step = 0

    def __iter__(self) -> DataLoader:  # type: ignore
        # mypy gets angry that the return type
        return self

    # def shutdown(self) -> None:
    #     with nostdout():
    #         if self.nnunet_dataloader is not None:
    #             if isinstance(self.nnunet_dataloader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
    #                 self.nnunet_dataloader._finish()


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
        """
        _summary_

        Args:
            dataset_id (int): _description_
            data_identifier (Optional[str]): _description_
            plans_identifier (Optional[str]): _description_
            always_preprocess (bool, optional): _description_. Defaults to True.
        """
        super().__init__(**kwargs)
        self.dataset_id: int = dataset_id
        self.dataset_name = convert_id_to_dataset_name(self.dataset_id)
        self.dataset_json = load_json(join(nnUNet_raw, self.dataset_name, "dataset.json"))
        self.data_identifier = data_identifier
        self.device = kwargs["device"]
        self.always_preprocess: bool = always_preprocess
        self.plans_name = plans_identifier

        self.nnunet_trainer: nnUNetTrainer

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        """
        Gets the nnunet dataloaders and wraps them in another class to make them
        pytorch iterators

        Args:
            config (Config): The config file from the server

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple of length two. The client
                train and validation dataloaders as pytorch dataloaders
        """
        # flwr 1.9.0 now raises an error any time a process ends.
        # To prevent errors due to this since dataloaders use multiprocessing
        # lets ovverride that behaviour before the child processes are created
        fl_sigint_handler = signal.getsignal(signal.SIGINT)
        fl_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, og_sigint_handler)
        signal.signal(signal.SIGTERM, og_sigterm_handler)

        # Get the nnunet dataloader iterators
        with nostdout():
            train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Set the signal handlers back to what they were for flwr
        signal.signal(signal.SIGINT, fl_sigint_handler)
        signal.signal(signal.SIGTERM, fl_sigterm_handler)

        # The batchgenerators package used under the hood by the dataloaders
        # creates an additional stream handler for the root logger
        # Therefore all logs get printed twice, We can fix this by clearing the
        # root logger handlers.
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # By default nnunet dataloaders are infinite, so if we are training by
        # num epochs we need to make them not infinite anymore
        if config["local_epochs"] is not None:
            infinite = False

        assert isinstance(self.nnunet_config, str)
        assert is_valid_config(self.nnunet_config)
        train_loader = nnUNetDLWrapper(
            nnunet_dataloader=train_loader, nnunet_config=self.nnunet_config, infinite=infinite
        )
        val_loader = nnUNetDLWrapper(nnunet_dataloader=val_loader, nnunet_config=self.nnunet_config, infinite=infinite)

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

    def maybe_preprocess(self, nnunet_config: nnUNetConfig) -> None:
        """
        Checks if preprocessed data for current plans exists and if not preprocesses the nnunet_raw dataset

        Args:
            nnunet_config (nnUNetConfig): string specifying the nnunet config
        """
        assert self.data_identifier is not None, "Was expecting data identifier to be initialized in self.create_plans"

        # Preprocess data if it's not already there
        if self.always_preprocess or not exists(self.nnunet_trainer.preprocessed_dataset_folder):
            default_num_processes = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
            if nnunet_config in default_num_processes:
                num_processes = default_num_processes[nnunet_config]
            else:
                num_processes = 4
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_name,
                num_processes=[
                    num_processes
                ],  # Since we only run one config at a time, we know num_processes is an int, change it to a list
                configurations=[nnunet_config],
            )
        else:
            log(INFO, "nnunet preprocessed data seems to already exist. Skipping preprocessing")

    def setup_client(self, config: Config) -> None:
        """
        Ensures the necessary files for training are on disk and initializes
        several class attributes that depend on values in the config from the
        server. This is called at the beginning of each fit round and each evaluate round

        Args:
            config (Config): The config file from the server. The nnUNetClient
                expects the keys 'nnunet_config', 'nnunet_plans', and 'fold' in
                addition to those required by BasicClient
        """
        log(INFO, "Setting up the nnUNetClient")

        self.nnunet_config = config["nnunet_config"]  # Need this for predict
        assert isinstance(self.nnunet_config, str)
        assert is_valid_config(self.nnunet_config)

        # Check if dataset fingerprint has been extracted
        if not exists(join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json")):
            log(INFO, "Extracting nnunet dataset fingerprint")
            with nostdout():
                extract_fingerprints(dataset_ids=[self.dataset_id])
        else:
            log(INFO, "nnunet dataset fingerprint already exists")

        # Get the dataset json and dataset name of the local client dataset
        self.plans = self.create_plans(config=config)
        with nostdout():
            # Create the nnunet trainer
            self.nnunet_trainer = nnUNetTrainer(
                plans=self.plans,
                configuration=self.nnunet_config,
                fold=config["fold"],
                dataset_json=self.dataset_json,
                device=self.device,
            )

            # nnunet_trainer initialization
            self.nnunet_trainer.initialize()
            # This is done by nnunet_trainer in self.on_train_start, we
            # do it manually since nnunet_trainer not being used for training
            self.nnunet_trainer.set_deep_supervision_enabled(self.nnunet_trainer.enable_deep_supervision)

        # Preprocess nnunet_raw data if needed
        self.maybe_preprocess(self.nnunet_config)
        unpack_dataset(  # Reduces load on CPU and RAM during training
            folder=self.nnunet_trainer.preprocessed_dataset_folder,
            unpack_segmentation=self.nnunet_trainer.unpack_dataset,
            overwrite_existing=self.always_preprocess,
            verify_npy=True,
        )  # Takes about 3 seconds for a small dataset of 24 samples

        # Parent function sets up optimizer, criterion, parameter_exchanger, dataloaders and reporters.
        # We have to run this at the end since it depends on the previous setup
        super().setup_client(config)

    def predict(self, input: TorchInputType) -> Tuple[TorchPredType, Dict[str, torch.Tensor]]:
        """
        Generate model outputs. Overridden because nnunets output lists when
        deep supervision is on so we have to reformat output into dicts

        Args:
            input (TorchInputType): The model inputs

        Returns:
            Tuple[TorchPredType, Dict[str, torch.Tensor]]: A tuple in which the
            first element model outputs indexed by name. The second element is
            unused by this subclass and therefore is always an empty dict
        """
        if isinstance(input, torch.Tensor):
            output = self.model(input)
        else:
            raise TypeError('"input" must be of type torch.Tensor for nnUNetClient')

        if isinstance(output, torch.Tensor):
            return {"prediction": output}, {}
        elif isinstance(output, (list, tuple)):
            assert isinstance(self.nnunet_config, str)
            assert is_valid_config(self.nnunet_config)
            num_spatial_dims = get_num_spatial_dims(self.nnunet_config)
            preds = convert_ds_list_to_dict(output, num_spatial_dims)
            return preds, {}
        else:
            raise TypeError(
                "Was expecting nnunet model output to be either a torch.Tensor or a list/tuple of torch.Tensors"
            )

    def compute_loss_and_additional_losses(
        self, preds: TorchPredType, features: Dict[str, torch.Tensor], target: TorchTargetType
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Checks the pred and target types and computes the loss

        Args:
            preds (TorchPredType): Dictionary of model output tensors indexed
                by name
            features (Dict[str, torch.Tensor]): Not used by this subclass
            target (TorchTargetType): The targets to evaluate the predictions
                with. If multiple prediction tensors are given, target must be
                a dictionary with the same number of tensors

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]: A tuple
                where the first element is the loss and the second element is an
                optional additional loss
        """
        # Check if deep supervision is on by checking the number of items in the pred and target dictionaries
        loss_preds: torch.Tensor | List[torch.Tensor]
        loss_targets: torch.Tensor | List[torch.Tensor]
        if len(preds) > 1:
            loss_preds = convert_ds_dict_to_list(preds)
        else:
            loss_preds = list(preds.values())[0]

        if isinstance(target, torch.Tensor):
            assert len(preds) == 1, "Got 1 target but possibly more than one prediction"
            loss_targets = target
        elif isinstance(target, dict):
            assert len(target) == len(
                preds
            ), "Was expecting the same number \
                of predictions and targets"
            if len(target) > 1:
                loss_targets = convert_ds_dict_to_list(target)
            else:
                loss_targets = list(target.values())[0]

        else:
            raise TypeError("Was expecting target to be type Dict[str, torch.Tensor] or torch.Tensor")

        return self.criterion(loss_preds, loss_targets), None

    def mask_data(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masks the pred and target tensors according to nnunet ignore_label.
        The number of classes in the input tensors should be at least 3
        corresponding to 2 classes for binary segmentation and 1 class which is
        the ignore class specified by ignore label. See nnunet documentation
        for more info

        Args:
            pred (torch.Tensor): The one hot encoded predicted
                segmentation maps with shape (batch, classes, x, y(, z))
            target (torch.Tensor): The ground truth segmentation map with shape
                (batch, classes, x, y(, z))

        Returns:
            torch.Tensor: The masked ohe predicted segmentation maps
            torch.Tensor: The masked target segmentation maps
        """
        # create mask where 1 is where pixels in target are not ignore label
        # Modify target to remove the last class which is the ignore_label class
        new_target = target
        if self.nnunet_trainer.label_manager.has_regions:  # nnunet returns a ohe target is has_regions is true
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]  # omit the last class, ie the ignore_label
            else:
                mask = 1 - target[:, -1:]
            new_target = new_target[:, :-1]  # Remove final ignore_label class from target
        else:  # target is not one hot encoded
            mask = (target != self.nnunet_trainer.label_manager.ignore_label).float()
            # Set ignore label to background essentially removing it as a class
            new_target[new_target == self.nnunet_trainer.label_manager.ignore_label] = 0

        # Tile the mask to be one hot encoded
        mask_here = torch.tile(mask, (1, pred.shape[1], *[1 for _ in range(2, pred.ndim)]))

        return pred * mask_here, new_target  # Mask the input tensor and return the modified target

    def update_metric_manager(
        self, preds: TorchPredType, target: TorchTargetType, metric_manager: MetricManager
    ) -> None:
        """
        Update the metrics with preds and target. Ovverriden because we might
        need to manipulate inputs due to deep supervision

        Args:
            preds (TorchPredType): dictionary of model outputs
            target (TorchTargetType): the targets generated by the dataloader
                to evaluate the preds with
            metric_manager (MetricManager): the metric manager to update
        """
        if len(preds) > 1:
            # for nnunet models the first pred in the output list is the main one
            m_pred = convert_ds_dict_to_list(preds)[0]

        if isinstance(target, torch.Tensor):
            m_target = target
        elif isinstance(target, dict):
            if len(target) > 1:
                m_target = convert_ds_dict_to_list(target)[0]
            else:
                m_target = list(target.values())[0]
        else:
            raise TypeError("Was expecting target to be type Dict[str, torch.Tensor] or torch.Tensor")

        # Check if target is one hot encoded.
        # Prediction always is for nnunet models
        # Add channel dimension if there isn't one
        if m_pred.ndim != m_target.ndim:
            m_target = m_target.view(m_target.shape[0], 1, *m_target.shape[1:])

        # One hot encode targets if needed
        if m_pred.shape != m_target.shape:
            m_target_ohe = torch.zeros(m_pred.shape, device=self.device, dtype=torch.bool)
            # This is how nnunet does ohe in their functions
            # Its a weird function that is not intuitive
            # CAREFUL: Notice the underscore at the end of the scatter function.
            # It makes a difference, was a hard bug to find!
            m_target_ohe.scatter_(1, m_target.long(), 1)

        # Check if ignore label is in use. The nnunet loss figures this out on
        # it's own, but we do it it manually here for the metrics
        if self.nnunet_trainer.label_manager.ignore_label is not None:
            m_pred, m_target_ohe = self.mask_data(m_pred, m_target_ohe)

        # m_pred is one hot encoded output logits. Maybe masked by ignore label
        # m_target_ohe is one hot encoded boolean label. Maybe masked by ignore label
        metric_manager.update({"prediction": m_pred}, m_target_ohe)
