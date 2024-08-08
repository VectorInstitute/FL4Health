import contextlib
import contextlib
import logging
import os
import pickle
import signal
import warnings
from logging import INFO
from os.path import exists, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar
from numpy import ceil
from flwr.common.logger import log
from flwr.common.typing import Config
from numpy import ceil
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, LoggingMode
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.config import narrow_config_type
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric, MetricManager
from fl4health.utils.typing import LogLevel, TorchInputType, TorchPredType, TorchTargetType
from research.picai.fl_nnunet.nnunet_utils import (
    Module2LossWrapper,
    NnUNetConfig,
    convert_deepsupervision_dict_to_list,
    convert_deepsupervision_list_to_dict,
    get_valid_nnunet_config,
    nnUNetDataLoaderWrapper,
)

with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet. https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
    from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.training.dataloading.utils import unpack_dataset
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


# Get the default signal handlers used by python before flwr overrides them
# We need these because the nnunet dataloaders spawn child processes
# and flwr throws errors when those processes end. So we set the signal handlers
# for the child processes to the python defaults to avoid this
ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
ORIGINAL_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)


class nnUNetClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        device: torch.device,
        dataset_id: int,
        fold: Union[int, str],
        data_identifier: Optional[str] = None,
        plans_identifier: Optional[str] = None,
        always_preprocess: bool = False,
        metrics: Optional[Sequence[Metric]] = None,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
        progress_bar: bool = False,
    ) -> None:
        """
        A client for training nnunet models. Requires the following additional
        keys in the config sent from the server
            'nnunet_plans': (serialized dict)
            'nnunet_config': (str)

        Args:
            data_path (Path): Required by the parent class but not used by
                nnUNetClient since it uses the nnunet environment variables
                to determine the data location.
            device (torch.device): Device indicator for where to send the
                model, batches, labels etc. Often 'cpu' or 'cuda' or 'mps'
            dataset_id (int): The nnunet dataset id for the local client dataset
                to use for training and validation.
            fold (Union[int, str]): Which fold of the local client dataset to
                use for validation. nnunet defaults to 5 folds (0 to 4). Can
                also be set to 'all' to use all the data for both training
                and validation.
            data_identifier (Optional[str], optional): The nnunet data
                identifier prefix to use. The final data identifier will be
                {data_identifier}_config where 'config' is the nnunet config
                (eg. 2d, 3d_fullres, etc.). If preprocessed data already exists
                can be used to specify which preprocessed data to use. The
                default data_identifier prefix is the plans name used during
                training (see the plans_identifier argument).
            plans_identifier (Optional[str], optional): Specify what to save
                the client's local copy of the plans file as. The client
                modifies the source plans json file sent from the server and
                makes a local copy. If left as default None, the plans
                identifier will be set as 'FL_Dataset000_plansname' where 000
                is the dataset_id and plansname is the 'plans_name' value of
                the source plans file.
            always_preprocess (bool, optional): If True, will preprocess the
                local client dataset even if the preprocessed data already
                seems to exist. Defaults to False. The existence of the
                preprocessed data is determined by matching the provided
                data_identifier with that of the preprocessed data already on
                the client.
            metrics (Sequence[Metric], optional): Metrics to be computed based
                on the labels and predictions of the client model. Defaults to [].
            loss_meter_type (LossMeterType, optional): Type of meter used to
                track and compute the losses over each batch. Defaults to
                LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional):
                Checkpointer module defining when and how to do checkpointing
                during client-side training. No checkpointing is done if not
                provided. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics
                reporter instance to record the metrics during the execution.
                Defaults to an instance of MetricsReporter with default init parameters.
        """
        metrics = metrics if metrics else []
        # Parent method sets up several class attributes
        super().__init__(
            data_path=data_path,  # self.data_path, not used by nnUNetClient
            metrics=metrics,  # self.metrics
            device=device,  # self.device
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,  # self.checkpointer
            metrics_reporter=metrics_reporter,  # self.metrics_reporter
            progress_bar=progress_bar,
        )

        # Some nnunet specific attributes
        self.dataset_id: int = dataset_id
        self.dataset_name = convert_id_to_dataset_name(self.dataset_id)
        self.dataset_json = load_json(join(nnUNet_raw, self.dataset_name, "dataset.json"))
        self.fold = fold
        self.data_identifier = data_identifier
        self.always_preprocess: bool = always_preprocess
        self.plans_name = plans_identifier
        self.fingerprint_extracted = False

        # nnunet specific attributes to be initialized in setup_client
        self.nnunet_trainer: nnUNetTrainer
        self.nnunet_config: NnUNetConfig
        self.plans: dict

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
        # lets overide that behaviour before the child processes are created
        fl_sigint_handler = signal.getsignal(signal.SIGINT)
        fl_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
        signal.signal(signal.SIGTERM, ORIGINAL_SIGTERM_HANDLER)

        # Get the nnunet dataloader iterators
        with contextlib.redirect_stdout(None):
            train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Set the signal handlers back to what they were for flwr
        signal.signal(signal.SIGINT, fl_sigint_handler)
        signal.signal(signal.SIGTERM, fl_sigterm_handler)

        # The batchgenerators package used under the hood by the dataloaders
        # creates an additional stream handler for the root logger
        # Therefore all logs get printed twice, We can fix this by clearing the
        # root logger handlers.
        # Issue: https://github.com/MIC-DKFZ/batchgenerators/issues/123
        # PR: https://github.com/MIC-DKFZ/batchgenerators/pull/124
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        train_loader = nnUNetDataLoaderWrapper(nnunet_augmenter=train_loader, nnunet_config=self.nnunet_config)
        val_loader = nnUNetDataLoaderWrapper(nnunet_augmenter=val_loader, nnunet_config=self.nnunet_config)

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
        plans = pickle.loads(narrow_config_type(config, "nnunet_plans", bytes))

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
            # Possible extension could be to increase batch size if possible
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

    def maybe_preprocess(self, nnunet_config: NnUNetConfig) -> None:
        """
        Checks if preprocessed data for current plans exists and if not
        preprocesses the nnunet_raw dataset. The preprocessed data is saved in
        '{nnUNet_preprocessed}/{dataset_name}/{data_identifier} where
        nnUNet_preprocessed is the directory specified by the
        nnUNet_preprocessed environment variable. dataset_name is the nnunet
        dataset name (eg. Dataset123_MyDataset) and data_identifier
        is {self.data_identifier}_{self.nnunet_config}

        Args:
            nnunet_config (NnUNetConfig): The nnunet config as a NnUNetConfig
                Enum. Enum type ensures nnunet config is valid
        """
        assert self.data_identifier is not None, "Was expecting data identifier to be initialized in self.create_plans"

        # Preprocess data if it's not already there
        if self.always_preprocess or not exists(self.nnunet_trainer.preprocessed_dataset_folder):
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_name,
                num_processes=[nnunet_config.default_num_processes],
                configurations=[nnunet_config.value],
            )
        else:
            log(INFO, "nnunet preprocessed data seems to already exist. Skipping preprocessing")

    def maybe_extract_fingerprint(self) -> None:
        """
        Checks if nnunet dataset fingerprint already exists and if not extracts one from the dataset
        """
        fp_path = join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json")
        if self.always_preprocess or not exists(fp_path):
            log(INFO, "\tExtracting nnunet dataset fingerprint")
            with contextlib.redirect_stdout(None):  # prevent print statements from nnunet method
                extract_fingerprints(dataset_ids=[self.dataset_id])
        else:
            log(INFO, "\tnnunet dataset fingerprint already exists. Skipping fingerprint extraction")

        # Avoid extracting fingerprint multiple times when always_preprocess is true
        self.fingerprint_extracted = True

    def setup_client(self, config: Config) -> None:
        """
        Ensures the necessary files for training are on disk and initializes
        several class attributes that depend on values in the config from the
        server. This is called once when the client is sampled by the server
        for the first time.

        Args:
            config (Config): The config file from the server. The nnUNetClient
                expects the keys 'nnunet_config' and 'nnunet_plans' in
                addition to those required by BasicClient
        """
        log(INFO, "Setting up the nnUNetClient")

        # Empty gpu cache because nnunet does it
        self.empty_cache()

        # Get nnunet config
        self.nnunet_config = get_valid_nnunet_config(narrow_config_type(config, "nnunet_config", str))

        # Check if dataset fingerprint has been extracted
        # Check if dataset fingerprint has already been extracted
        if not self.fingerprint_extracted:
            self.maybe_extract_fingerprint()
        else:
            log(INFO, "\tDataset fingerprint has already been extracted. Skipping.")

        # Create the nnunet plans for the local client
        self.plans = self.create_plans(config=config)
        local_epochs, local_steps, _, _ = self.process_config(config)
        with contextlib.redirect_stdout(None):  # prevent print statements from nnunet methods
            # Create the nnunet trainer
            self.nnunet_trainer = nnUNetTrainer(
                plans=self.plans,
                configuration=self.nnunet_config.value,
                fold=self.fold,
                dataset_json=self.dataset_json,
                device=self.device,
            )

            # Need to modify num_epochs before initializing so that LRScheduler
            # recieves the right number of epochs
            # This will anneal LR to near zero each round
            # Default nnunet behaviour is do decrease LR every 250 steps
            # Maybe LRScheduler should just be an argument
            local_epochs, local_steps, _, _ = self.process_config(config)
            if local_steps is not None:
                num_samples = self.dataset_json["numTraining"]
                batch_size = self.plans["configurations"][self.nnunet_config.value]["batch_size"]
                steps_per_epoch = ceil(num_samples / batch_size)
                local_epochs = int(local_steps / steps_per_epoch)
            self.nnunet_trainer.num_epochs = local_epochs
            # nnunet_trainer initialization
            self.nnunet_trainer.initialize()
            # This is done by nnunet_trainer in self.on_train_start, we
            # do it manually since nnunet_trainer not being used for training
            self.nnunet_trainer.set_deep_supervision_enabled(self.nnunet_trainer.enable_deep_supervision)

        # Prevent nnunet from generating log files. And delete empty output directories
        os.remove(self.nnunet_trainer.log_file)
        self.nnunet_trainer.log_file = os.devnull
        output_folder = Path(self.nnunet_trainer.output_folder)
        while True:
            if len(os.listdir(output_folder)) == 0:
                os.rmdir(output_folder)
                output_folder = output_folder.parent
            else:
                break

        # Preprocess nnunet_raw data if needed
        self.maybe_preprocess(self.nnunet_config)
        unpack_dataset(  # Reduces load on CPU and RAM during training
            folder=self.nnunet_trainer.preprocessed_dataset_folder,
            unpack_segmentation=self.nnunet_trainer.unpack_dataset,
            overwrite_existing=self.always_preprocess,
            verify_npy=True,
        )  # Takes about 3 seconds for a small dataset of 24 samples

        # Parent function sets up optimizer, criterion, parameter_exchanger, dataloaders and reporters.
        # We have to run this at the end since it depends on the nnunet_trainer
        super().setup_client(config)
        log(INFO, "nnUNetClient Setup Complete")

    def predict(self, input: TorchInputType) -> Tuple[TorchPredType, Dict[str, torch.Tensor]]:
        """
        Generate model outputs. Overridden because nnunet's output lists when
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
            num_spatial_dims = self.nnunet_config.num_spatial_dims
            preds = convert_deepsupervision_list_to_dict(output, num_spatial_dims)
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

        def prepare_loss_arg(
            tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]
        ) -> Union[torch.Tensor, List[torch.Tensor]]:
            if isinstance(tensor, torch.Tensor):
                return tensor
            elif isinstance(tensor, dict):
                if len(tensor) > 1:
                    return convert_deepsupervision_dict_to_list(tensor)
                else:
                    return list(tensor.values())[0]

        loss_preds = prepare_loss_arg(preds)
        loss_targets = prepare_loss_arg(target)

        # Ensure we have the same number of predictions and targets
        assert isinstance(
            loss_preds, type(loss_targets)
        ), f"Got unexpected types for preds and targets: {type(loss_preds)} and {type(loss_targets)}"

        if isinstance(loss_preds, list):
            assert len(loss_preds) == len(
                loss_targets
            ), f"""Was expecting the number of predictions and targets to be
                the same. Got {len(loss_preds)} predictions and
                {len(loss_targets)} targets"""

        return self.criterion(loss_preds, loss_targets), None

    def mask_data(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masks the pred and target tensors according to nnunet ignore_label.
        The number of classes in the input tensors should be at least 3
        corresponding to 2 classes for binary segmentation and 1 class which is
        the ignore class specified by ignore label. See nnunet documentation
        for more info:
        https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/ignore_label.md

        Args:
            pred (torch.Tensor): The one hot encoded predicted
                segmentation maps with shape (batch, classes, x, y(, z))
            target (torch.Tensor): The ground truth segmentation map with shape
                (batch, classes, x, y(, z))

        Returns:
            torch.Tensor: The masked one hot encoded predicted segmentation maps
            torch.Tensor: The masked target segmentation maps
        """
        # create mask where 1 is where pixels in target are not ignore label
        # Modify target to remove the last class which is the ignore_label class
        new_target = target
        if self.nnunet_trainer.label_manager.has_regions:  # nnunet returns a ohe target if has_regions is true
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
        Update the metrics with preds and target. Overridden because we might
        need to manipulate inputs due to deep supervision

        Args:
            preds (TorchPredType): dictionary of model outputs
            target (TorchTargetType): the targets generated by the dataloader
                to evaluate the preds with
            metric_manager (MetricManager): the metric manager to update
        """
        if len(preds) > 1:
            # for nnunet the first pred in the output list is the main one
            m_pred = convert_deepsupervision_dict_to_list(preds)[0]

        if isinstance(target, torch.Tensor):
            m_target = target
        elif isinstance(target, dict):
            if len(target) > 1:
                # If deep supervision is in use, we drop the additional targets
                # when calculating the metrics as we only care about the
                # original target which by default in nnunet is at index 0
                m_target = convert_deepsupervision_dict_to_list(target)[0]
            else:
                m_target = list(target.values())[0]
        else:
            raise TypeError("Was expecting target to be type Dict[str, torch.Tensor] or torch.Tensor")

        # Check if target is one hot encoded. Prediction always is for nnunet
        # Add channel dimension if there isn't one
        if m_pred.ndim != m_target.ndim:
            m_target = m_target.view(m_target.shape[0], 1, *m_target.shape[1:])

        # One hot encode targets if needed
        if m_pred.shape != m_target.shape:
            m_target_one_hot = torch.zeros(m_pred.shape, device=self.device, dtype=torch.bool)
            # This is how nnunet does ohe in their functions
            # Its a weird function that is not intuitive
            # CAREFUL: Notice the underscore at the end of the scatter function.
            # It makes a difference, was a hard bug to find!
            m_target_one_hot.scatter_(1, m_target.long(), 1)
        else:
            m_target_one_hot = m_target

        # Check if ignore label is in use. The nnunet loss figures this out on
        # it's own, but we do it it manually here for the metrics
        if self.nnunet_trainer.label_manager.ignore_label is not None:
            m_pred, m_target_one_hot = self.mask_data(m_pred, m_target_one_hot)

        # m_pred is one hot encoded output logits. Maybe masked by ignore label
        # m_target_one_hot is one hot encoded boolean label. Maybe masked by ignore label
        metric_manager.update({"prediction": m_pred}, m_target_one_hot)

    def empty_cache(self) -> None:
        """
        Checks torch device and empties cache before training to optimize VRAM usage
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        else:
            pass

    def update_before_train(self, current_server_round: int) -> None:
        """
        Reset LR at beginning of training so that initial log str has correct LR and not the LR from the previous round
        """
        #
        self.nnunet_trainer.lr_scheduler.step(0)

    def update_before_step(self, step: int) -> None:
        """
        Update the learning rate. Need to define this in case train by steps is being used
        """
        # By default nnunet lr schedulers use epochs
        current_epoch = int((step) / len(self.train_loader))
        self.nnunet_trainer.lr_scheduler.step(current_epoch)

    def get_client_specific_logs(
        self, current_round: Optional[int], current_epoch: Optional[int], logging_mode: LoggingMode
    ) -> Tuple[str, List[Tuple[LogLevel, str]]]:
        if logging_mode == LoggingMode.TRAIN:
            lr = self.optimizers["global"].param_groups[0]["lr"]
            if current_epoch is None:
                # Assume training by steps
                return f"Initial LR {lr}", []
            else:
                return f" Current LR: {lr}", []
        else:
            return "", []

    def update_before_epoch(self, epoch: int) -> None:
        """
        Updates the learning rate at the beginning of the epoch. Technically
        LR is already being updated in self.update_before_step, but if
        training by epochs, we need to do it here too or the initial log
        string will have the incorrect LR.
        """

        self.nnunet_trainer.lr_scheduler.step(epoch)

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties (sample counts and nnunet plans) of client.

        If nnunet plans are not provided by the server, creates a new set of
        nnunet plans from the local client dataset. These plans are intended
        to be used for initializing global nnunet plans when they are not
        provided.

        Args:
            config (Config): The config from the server

        Returns:
            Dict[str, Scalar]: A dictionary containing the train and
                validation sample counts as well as the serialized nnunet plans
        """
        # Check if nnunet plans have already been initialized
        if "nnunet_plans" in config.keys():
            properties = super().get_properties(config)
            properties["nnunet_plans"] = config["nnunet_plans"]
            return properties

        # Check if local nnunet dataset fingerprint needs to be extracted
        if not self.fingerprint_extracted:
            self.maybe_extract_fingerprint()

        # Create experiment planner and plans
        planner = ExperimentPlanner(dataset_name_or_id=self.dataset_id, plans_name="temp_plans")
        with contextlib.redirect_stdout(None):  # Prevent print statements from experiment planner
            plans = planner.plan_experiment()
        plans_bytes = pickle.dumps(plans)

        # Remove plans file that was created by planner
        plans_path = join(nnUNet_preprocessed, self.dataset_name, planner.plans_identifier + ".json")
        if exists(plans_path):
            os.remove(plans_path)

        # return properties with initialized nnunet plans. Need to provide
        # plans since client needs to be initialized to get properties
        config["nnunet_plans"] = plans_bytes
        properties = super().get_properties(config)
        properties["nnunet_plans"] = pickle.dumps(plans_bytes)
        return properties

    def shutdown_dataloader(self, dataloader: Optional[DataLoader], dl_name: Optional[str] = None) -> None:
        """
        Checks the dataloaders type and if it is a MultiThreadedAugmenter or
        NonDetMultiThreadedAugmenter calls the _finish method to ensure they
        are properly shutdown

        Args:
            dataloader (DataLoader): The dataloader to shutdown
            dl_name (Optional[str]): A string that identifies the dataloader
                to shutdown. Used for logging purposes. Defaults to None
        """
        if dataloader is not None and isinstance(dataloader, nnUNetDataLoaderWrapper):
            if isinstance(dataloader.nnunet_dataloader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                if dl_name is not None:
                    log(INFO, f"\tShutting down nnunet dataloader: {dl_name}")
                dataloader.nnunet_dataloader._finish()

    def shutdown(self) -> None:
        # Not entirely sure if processes potentially opened by nnunet
        # dataloaders were being ended so ensure that they are ended here
        self.shutdown_dataloader(self.train_loader, "train_loader")
        self.shutdown_dataloader(self.val_loader, "val_loader")
        self.shutdown_dataloader(self.test_loader, "test_loader")
        return super().shutdown()
