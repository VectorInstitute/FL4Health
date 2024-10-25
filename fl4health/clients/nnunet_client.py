import gc
import logging
import os
import pickle
import time
import warnings
from contextlib import redirect_stdout
from logging import DEBUG, INFO, WARNING
from os.path import exists, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from flwr.common.logger import FLOWER_LOGGER, console_handler, log
from flwr.common.typing import Config, Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, LoggingMode
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric, MetricManager
from fl4health.utils.nnunet_utils import (
    NNUNET_DEFAULT_NP,
    NNUNET_N_SPATIAL_DIMS,
    Module2LossWrapper,
    NnunetConfig,
    PolyLRSchedulerWrapper,
    StreamToLogger,
    convert_deep_supervision_dict_to_list,
    convert_deep_supervision_list_to_dict,
    get_dataset_n_voxels,
    nnUNetDataLoaderWrapper,
    prepare_loss_arg,
    use_default_signal_handlers,
)
from fl4health.utils.typing import LogLevel, TorchInputType, TorchPredType, TorchTargetType

with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet. https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.training.dataloading.utils import unpack_dataset
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name


class NnunetClient(BasicClient):
    def __init__(
        self,
        device: torch.device,
        dataset_id: int,
        fold: Union[int, str],
        data_identifier: Optional[str] = None,
        plans_identifier: Optional[str] = None,
        compile: bool = True,
        always_preprocess: bool = False,
        max_grad_norm: float = 12,
        n_dataload_processes: Optional[int] = None,
        verbose: bool = True,
        metrics: Optional[Sequence[Metric]] = None,
        progress_bar: bool = False,
        intermediate_client_state_dir: Optional[Path] = None,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        client_name: Optional[str] = None,
    ) -> None:
        """
        A client for training nnunet models. Requires the nnunet environment variables
        to be set. Also requires the following additional keys in the config sent from
        the server
            'nnunet_plans': (serialized dict)
            'nnunet_config': (str)

        Args:
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
            compile (bool, optional): If True, the client will jit compile the pytorch
                model. This requires some overhead time at the beginning of training to
                compile the model, but results in faster training times. Defaults to
                True
            always_preprocess (bool, optional): If True, will preprocess the
                local client dataset even if the preprocessed data already
                seems to exist. Defaults to False. The existence of the
                preprocessed data is determined by matching the provided
                data_identifier with that of the preprocessed data already on
                the client.
            max_grad_norm (float, optional): The maximum gradient norm to use for
                gradient clipping. Defaults to 12 which is the nnunetv2 2.5.1 default.
            n_dataload_processes (Optional[int], optional): The number of processes to
                spawn for each nnunet dataloader. If left as None we use the nnunetv2
                version 2.5.1 defaults for each config
            verbose (bool, optional): If True the client will log some extra INFO logs.
                Defaults to False unless the log level is DEBUG or lower.
            metrics (Sequence[Metric], optional): Metrics to be computed based
                on the labels and predictions of the client model. Defaults to [].
            progress_bar (bool, optional): Whether or not to print a progress bar to
                stdout for training. Defaults to False
            intermediate_client_state_dir (Optional[Path]): An optional path to store per round
                checkpoints.
            loss_meter_type (LossMeterType, optional): Type of meter used to
                track and compute the losses over each batch. Defaults to
                LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional):
                Checkpointer module defining when and how to do checkpointing
                during client-side training. No checkpointing is done if not
                provided. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
        """
        metrics = metrics if metrics else []
        # Parent method sets up several class attributes
        super().__init__(
            data_path=Path("dummy/path"),  # data_path not used by NnunetClient
            metrics=metrics,  # self.metrics
            device=device,  # self.device
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,  # self.checkpointer
            reporters=reporters,
            progress_bar=progress_bar,
            intermediate_client_state_dir=intermediate_client_state_dir,
            client_name=client_name,
        )

        # Some nnunet client specific attributes
        self.dataset_id: int = dataset_id
        self.dataset_name = convert_id_to_dataset_name(self.dataset_id)
        self.dataset_json = load_json(join(nnUNet_raw, self.dataset_name, "dataset.json"))
        self.fold = fold
        self.data_identifier = data_identifier
        self.always_preprocess = always_preprocess
        self.plans_name = plans_identifier
        self.fingerprint_extracted = False
        self.max_grad_norm = max_grad_norm
        self.n_dataload_proc = n_dataload_processes

        # Auto set verbose to True if console handler is on DEBUG mode
        self.verbose = verbose if console_handler.level >= INFO else True

        # Used to redirect stdout to logger
        self.stream2debug = StreamToLogger(FLOWER_LOGGER, DEBUG)

        # nnunet specific attributes to be initialized in setup_client
        self.nnunet_trainer: nnUNetTrainer
        self.nnunet_config: NnunetConfig
        self.plans: dict[str, Any]
        self.steps_per_round: int  # N steps per server round
        self.max_steps: int  # N_rounds x steps_per_round

        # Set nnunet compile environment variable. Nnunet default is to compile
        if not compile:
            if self.verbose:
                log(INFO, "Switching pytorch model jit compile to OFF")
            os.environ["nnUNet_compile"] = str("false")

    @use_default_signal_handlers  # Dataloaders use multiprocessing
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
        start_time = time.time()
        # Set the number of processes for each dataloader.
        if self.n_dataload_proc is None:
            # Nnunet default is 12 or max cpu's. We decrease max by 1 just in case
            # NOTE: The type: ignore here is to skip issues where a local operating system is not compatible
            # with sched_getaffinity (older versions of MacOS, for example). The code still won't run but mypy won't
            # complain. Workarounds like using os.cpu_count(), while not exactly the same, are possible.
            self.n_dataload_proc = min(12, len(os.sched_getaffinity(0)) - 1)  # type: ignore
        os.environ["nnUNet_n_proc_DA"] = str(self.n_dataload_proc)

        # The batchgenerators package used under the hood by the dataloaders creates an
        # additional stream handler for the root logger Therefore all logs get printed
        # twice. First we stop the flwr logger from passing logs to the root logger
        # Issue: https://github.com/MIC-DKFZ/batchgenerators/issues/123
        # PR: https://github.com/MIC-DKFZ/batchgenerators/pull/124
        FLOWER_LOGGER.propagate = False

        # Redirect nnunet output to flwr logger at DEBUG level
        with redirect_stdout(self.stream2debug):
            # Get the nnunet dataloader iterators. (Technically augmenter classes)
            train_loader, val_loader = self.nnunet_trainer.get_dataloaders()

        # Now clear the root handler that was create and turn propagate back to true
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        FLOWER_LOGGER.propagate = True

        # Wrap nnunet dataloaders to make them compatible with fl4health
        train_loader = nnUNetDataLoaderWrapper(nnunet_augmenter=train_loader, nnunet_config=self.nnunet_config)
        val_loader = nnUNetDataLoaderWrapper(nnunet_augmenter=val_loader, nnunet_config=self.nnunet_config)

        if self.verbose:
            log(INFO, f"\tDataloaders initialized in {time.time()-start_time:.1f}s")

        return train_loader, val_loader

    def get_model(self, config: Config) -> nn.Module:
        return self.nnunet_trainer.network

    def get_criterion(self, config: Config) -> _Loss:
        return Module2LossWrapper(self.nnunet_trainer.loss)

    def get_optimizer(self, config: Config) -> Optimizer:
        return self.nnunet_trainer.optimizer

    def get_lr_scheduler(self, optimizer_key: str, config: Config) -> _LRScheduler:
        """
        Creates an LR Scheduler similar to the nnunet default except we set max steps
        to the total number of steps and update every step. Initial and final LR are
        the same as nnunet, difference is nnunet sets max steps to num 'epochs', but
        they define an 'epoch' as exactly 250 steps. Therefore they update every 250
        steps. Override this method to set your own LR scheduler.

        Args:
            config (Config): The server config. This method will look for the
        Returns:
            _LRScheduler: The default nnunet LR Scheduler for nnunetv2 2.5.1
        """
        if not isinstance(self.nnunet_trainer.lr_scheduler, PolyLRScheduler):
            log(
                WARNING,
                (
                    "Nnunet seems to have changed their default LR scheduler to "
                    f"type: {type(self.nnunet_trainer.lr_scheduler)}. "
                    "Using PolyLRScheduler instead. Override or update the "
                    "get_lr_scheduler method of nnUNetClient to change this"
                ),
            )

        # Determine total number of steps throughout all FL rounds
        local_epochs, local_steps, _, _ = self.process_config(config)
        if local_steps is not None:
            steps_per_round = local_steps
        elif local_epochs is not None:
            steps_per_round = local_epochs * len(self.train_loader)
        else:
            raise ValueError("One of local steps or local epochs must be specified")

        total_steps = int(config["n_server_rounds"]) * steps_per_round

        # Create and return LR Scheduler Wrapper for the PolyLRScheduler so that it is
        # compatible with Torch LRScheduler
        # Create and return LR Scheduler. This is nnunet default for version 2.5.1
        return PolyLRSchedulerWrapper(
            self.optimizers[optimizer_key],
            initial_lr=self.nnunet_trainer.initial_lr,
            max_steps=total_steps,
        )

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
        plans = pickle.loads(narrow_dict_type(config, "nnunet_plans", bytes))

        # Change plans name.
        if self.plans_name is None:
            self.plans_name = "FL_source-" + plans["plans_name"]

        plans["source_plans_name"] = plans["plans_name"]
        plans["plans_name"] = self.plans_name

        # Change dataset name
        plans["dataset_name"] = self.dataset_name

        # Change data identifier
        if self.data_identifier is None:
            self.data_identifier = self.plans_name

        # Get maximum number of voxels for a batch based on dataset size
        # TODO: This function assumes the median image size of the local dataset is the
        # same as the one from which the plans file was created. Better way to do it is
        # to pass fingerprint as a param and compute median image size manually.
        n_cases = self.dataset_json["numTraining"]
        max_voxels = get_dataset_n_voxels(plans, n_cases) * 0.05  # Max is 5% of total

        # Iterate through nnunet configs in plans file
        for c in plans["configurations"].keys():
            # Change the data identifier
            plans["configurations"][c]["data_identifier"] = self.data_identifier + "_" + c
            # Ensure batch size is at least 2, then at most 5 percent of dataset
            # TODO: Possible extension could be to increase batch size if possible
            if "batch_size" in plans["configurations"][c].keys():
                old_bs = plans["configurations"][c]["batch_size"]
                bs_5percent = round(
                    (  # Patch size is the input shape to the model
                        max_voxels / np.prod(plans["configurations"][c]["patch_size"], dtype=np.float64)
                    )
                )
                new_bs = max(min(old_bs, bs_5percent), 2)
                plans["configurations"][c]["batch_size"] = new_bs
            else:
                log(
                    WARNING,
                    ("Did not find a 'batch_size' key in the nnunet plans " f"dict for nnunet config: {c}"),
                )

        # Can't run nnunet preprocessing without saving plans file
        os.makedirs(join(nnUNet_preprocessed, self.dataset_name), exist_ok=True)
        plans_save_path = join(nnUNet_preprocessed, self.dataset_name, self.plans_name + ".json")
        save_json(plans, plans_save_path, sort_keys=False)
        return plans

    @use_default_signal_handlers  # Preprocessing spawns subprocesses
    def maybe_preprocess(self, nnunet_config: NnunetConfig) -> None:
        """
        Checks if preprocessed data for current plans exists and if not
        preprocesses the nnunet_raw dataset. The preprocessed data is saved in
        '{nnUNet_preprocessed}/{dataset_name}/{data_identifier} where
        nnUNet_preprocessed is the directory specified by the
        nnUNet_preprocessed environment variable. dataset_name is the nnunet
        dataset name (eg. Dataset123_MyDataset) and data_identifier
        is {self.data_identifier}_{self.nnunet_config}

        Args:
            nnunet_config (NnunetConfig): The nnunet config as a NnunetConfig
                Enum. Enum type ensures nnunet config is valid
        """
        assert self.data_identifier is not None, "Was expecting data identifier to be initialized in self.create_plans"

        # Preprocess data if it's not already there
        if self.always_preprocess or not exists(self.nnunet_trainer.preprocessed_dataset_folder):
            if self.verbose:
                log(INFO, f"\tPreprocessing local client dataset: {self.dataset_name}")
            # Unless log level is debugging or lower, hide nnunet output
            with redirect_stdout(self.stream2debug):
                preprocess_dataset(
                    dataset_id=self.dataset_id,
                    plans_identifier=self.plans_name,
                    num_processes=[NNUNET_DEFAULT_NP[nnunet_config]],
                    configurations=[nnunet_config.value],
                )
        elif self.verbose:
            log(
                INFO,
                "\tnnunet preprocessed data seems to already exist. Skipping preprocessing",
            )

    @use_default_signal_handlers  # Fingerprint extraction spawns subprocess
    def maybe_extract_fingerprint(self) -> None:
        """
        Checks if nnunet dataset fingerprint already exists and if not extracts one from the dataset
        """
        # Check first whether this client instance has already extracted a dataset fp
        # Possible if the client was asked to generate the nnunet plans for the server
        if not self.fingerprint_extracted:
            fp_path = join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json")
            # Check if fp already exists or if we want to redo fp extraction
            if self.always_preprocess or not exists(fp_path):
                start = time.time()
                # Unless log level is DEBUG or lower hide nnunet output
                with redirect_stdout(self.stream2debug):
                    extract_fingerprints(dataset_ids=[self.dataset_id])
                if self.verbose:
                    log(
                        INFO,
                        f"\tExtracted dataset fingerprint in {time.time()-start:.1f}s",
                    )
            elif self.verbose:
                log(
                    INFO,
                    "\tnnunet dataset fingerprint already exists. Skipping fingerprint extraction",
                )
        elif self.verbose:
            log(
                INFO,
                "\tThis client instance has already extracted the dataset fingerprint. Skipping.",
            )

        # Avoid extracting fingerprint multiple times when always_preprocess is true
        self.fingerprint_extracted = True

    # Several subprocesses spawned in setup during torch.compile and dataset unpacking
    @use_default_signal_handlers
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
        self.nnunet_config = NnunetConfig(config["nnunet_config"])

        # Check if dataset fingerprint has already been extracted
        self.maybe_extract_fingerprint()

        # Create the nnunet plans for the local client
        self.plans = self.create_plans(config=config)

        # Unless log level is DEBUG or lower hide nnunet output
        with redirect_stdout(self.stream2debug):
            # Create the nnunet trainer
            self.nnunet_trainer = nnUNetTrainer(
                plans=self.plans,
                configuration=self.nnunet_config.value,
                fold=self.fold,
                dataset_json=self.dataset_json,
                device=self.device,
            )
            # nnunet_trainer initialization
            self.nnunet_trainer.initialize()
            # This is done by nnunet_trainer in self.on_train_start, we
            # do it manually since nnunet_trainer not being used for training
            self.nnunet_trainer.set_deep_supervision_enabled(self.nnunet_trainer.enable_deep_supervision)

        # Prevent nnunet from generating log files and delete empty output directories
        os.remove(self.nnunet_trainer.log_file)
        self.nnunet_trainer.log_file = os.devnull
        output_folder = Path(self.nnunet_trainer.output_folder)
        while len(os.listdir(output_folder)) == 0:
            os.rmdir(output_folder)
            output_folder = output_folder.parent

        # Preprocess nnunet_raw data if needed
        self.maybe_preprocess(self.nnunet_config)
        start = time.time()
        unpack_dataset(  # Reduces load on CPU and RAM during training
            folder=self.nnunet_trainer.preprocessed_dataset_folder,
            unpack_segmentation=self.nnunet_trainer.unpack_dataset,
            overwrite_existing=self.always_preprocess,
            verify_npy=True,
        )
        if self.verbose:
            log(INFO, f"\tUnpacked dataset in {time.time()-start:.1f}s")

        # We have to call parent method after setting up nnunet trainer
        super().setup_client(config)

    def predict(self, input: TorchInputType) -> Tuple[TorchPredType, Dict[str, torch.Tensor]]:
        """
        Generate model outputs. Overridden because nnunets output lists when
        deep supervision is on so we have to reformat the output into dicts

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
            num_spatial_dims = NNUNET_N_SPATIAL_DIMS[self.nnunet_config]
            preds = convert_deep_supervision_list_to_dict(output, num_spatial_dims)
            return preds, {}
        else:
            raise TypeError(
                "Was expecting nnunet model output to be either a torch.Tensor or a list/tuple of torch.Tensors"
            )

    def compute_loss_and_additional_losses(
        self,
        preds: TorchPredType,
        features: Dict[str, torch.Tensor],
        target: TorchTargetType,
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
        # prepare loss args check if deep supervision is on and returns list if so
        loss_preds = prepare_loss_arg(preds)
        loss_targets = prepare_loss_arg(target)

        # Ensure we have the same number of predictions and targets
        assert isinstance(
            loss_preds, type(loss_targets)
        ), f"Got unexpected types for preds and targets: {type(loss_preds)} and {type(loss_targets)}"

        if isinstance(loss_preds, list):
            assert len(loss_preds) == len(loss_targets), (
                "Was expecting the number of predictions and targets to be the same. "
                f"Got {len(loss_preds)} predictions and {len(loss_targets)} targets."
            )

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

        return (
            pred * mask_here,
            new_target,
        )  # Mask the input tensor and return the modified target

    def update_metric_manager(
        self,
        preds: TorchPredType,
        target: TorchTargetType,
        metric_manager: MetricManager,
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
            m_pred = convert_deep_supervision_dict_to_list(preds)[0]

        if isinstance(target, torch.Tensor):
            m_target = target
        elif isinstance(target, dict):
            if len(target) > 1:
                # If deep supervision is in use, we drop the additional targets
                # when calculating the metrics as we only care about the
                # original target which by default in nnunet is at index 0
                m_target = convert_deep_supervision_dict_to_list(target)[0]
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
        # it's own, but we do it manually here for the metrics
        if self.nnunet_trainer.label_manager.ignore_label is not None:
            m_pred, m_target_one_hot = self.mask_data(m_pred, m_target_one_hot)

        # m_pred is one hot encoded (OHE) output logits. Maybe masked by ignore label
        # m_target_one_hot is OHE boolean label. Maybe masked by ignore label
        metric_manager.update({"prediction": m_pred}, m_target_one_hot)

    def empty_cache(self) -> None:
        """
        Checks torch device and empties cache before training to optimize VRAM usage
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def get_client_specific_logs(
        self,
        current_round: Optional[int],
        current_epoch: Optional[int],
        logging_mode: LoggingMode,
    ) -> Tuple[str, List[Tuple[LogLevel, str]]]:
        if logging_mode == LoggingMode.TRAIN:
            lr = float(self.optimizers["global"].param_groups[0]["lr"])
            if current_epoch is None:
                # Assume training by steps
                return f"Initial LR {lr}", []
            else:
                return f" Current LR: {lr}", []
        else:
            return "", []

    @use_default_signal_handlers  # Experiment planner spawns a process I think
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
            properties["num_input_channels"] = self.nnunet_trainer.num_input_channels
            properties["num_segmentation_heads"] = self.nnunet_trainer.label_manager.num_segmentation_heads
            properties["enable_deep_supervision"] = self.nnunet_trainer.enable_deep_supervision
            return properties

        # Check if local nnunet dataset fingerprint needs to be extracted
        self.maybe_extract_fingerprint()

        # Create experiment planner and plans. Plans name must be temp_plans so that we
        # can safely delete the plans file generated by the experiment planner
        planner = ExperimentPlanner(dataset_name_or_id=self.dataset_id, plans_name="temp_plans")

        # Unless log level is DEBUG or lower hide nnunet output
        with redirect_stdout(self.stream2debug):
            plans = planner.plan_experiment()

        # Set plans name to local dataset so we know the source
        # Dataset name normally begins with Dataset123_, we remove it and keep suffix
        plans["plans_name"] = self.dataset_name[11:] + "_plans"
        plans_bytes = pickle.dumps(plans)

        # Remove plans file . A new one will be generated in self.setup_client
        plans_path = join(nnUNet_preprocessed, self.dataset_name, planner.plans_identifier + ".json")
        if exists(plans_path):
            os.remove(plans_path)

        # return properties with initialized nnunet plans. Need to provide
        # plans since client needs to be initialized to get properties
        config["nnunet_plans"] = plans_bytes
        properties = super().get_properties(config)
        properties["nnunet_plans"] = plans_bytes
        properties["num_input_channels"] = self.nnunet_trainer.num_input_channels
        properties["num_segmentation_heads"] = self.nnunet_trainer.label_manager.num_segmentation_heads
        properties["enable_deep_supervision"] = self.nnunet_trainer.enable_deep_supervision
        return properties

    def shutdown_dataloader(self, dataloader: Optional[DataLoader], dl_name: Optional[str] = None) -> None:
        """
        The nnunet dataloader/augmenter uses multiprocessing under the hood, so the
        shutdown method terminates the child processes gracefully

        Args:
            dataloader (DataLoader): The dataloader to shutdown
            dl_name (Optional[str]): A string that identifies the dataloader
                to shutdown. Used for logging purposes. Defaults to None
        """
        if dataloader is not None and isinstance(dataloader, nnUNetDataLoaderWrapper):
            if self.verbose:
                log(INFO, f"\tShutting down nnunet dataloader: {dl_name}")
            dataloader.shutdown()

        del dataloader

    def shutdown(self) -> None:
        # Unfreeze and collect memory that was frozen during training
        # See self.update_before_train()
        gc.unfreeze()
        gc.collect()

        # Shutdown dataloader subprocesses gracefully
        self.shutdown_dataloader(self.train_loader, "train_loader")
        self.shutdown_dataloader(self.val_loader, "val_loader")
        self.shutdown_dataloader(self.test_loader, "test_loader")

        # Parent shutdown
        super().shutdown()

    def update_before_train(self, current_server_round: int) -> None:
        # Was getting OOM errors that could only be fixed by manually cleaning up RAM
        # https://github.com/pytorch/pytorch/issues/95462
        # The above issue seems to be the situation I was in.
        gc.collect()  # Cleans up unused variables
        # As the linked issue above points out, calling gc.freeze() greatly reduces the
        # overhead of garbage collection. (from 1.5s to 0.005s)
        if current_server_round == 2:
            # Collect runs even faster if we freeze after the end of the first iteration
            # Likely because a lot of variables are created in the first pass. If we
            # freeze before the first pass, gc.collect has to check all those variables
            gc.freeze()

    def transform_gradients(self, losses: TrainingLosses) -> None:
        """
        Apply the gradient clipping performed by the default nnunet trainer. This is
        the default behaviour for nnunet 2.5.1
        """
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
