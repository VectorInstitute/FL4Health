import datetime
from collections.abc import Sequence
from enum import Enum
from logging import INFO
from pathlib import Path

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import CheckpointMode, ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.fedrep_base import FedRepModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.typing import TorchInputType, TorchPredType, TorchTargetType


EpochsAndStepsTuple = tuple[int | None, int | None, int | None, int | None]


class FedRepTrainMode(Enum):
    HEAD = "head"
    REPRESENTATION = "representation"


class FedRepClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        """
        Client implementing the training of FedRep (https://arxiv.org/abs/2303.05206).

        Similar to FedPer, FedRep trains a global feature extractor shared by all clients through FedAvg and a
        private classifier that is unique to each client. However, FedRep breaks up the client-side training of
        these components into two phases. First the local classifier is trained with the feature extractor frozen.
        Next, the classifier is frozen and the feature extractor is trained.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        self.fedrep_train_mode = FedRepTrainMode.HEAD

    def _prepare_train_representations(self) -> None:
        """
        Handles the components switching needed to train the representation submodule as required by FedRep.

        This includes:

        1. Setting the training mode enum to know which optimizer should be stepping during training
        2. Unfreezing the base module, which represents the feature extractor (if frozen)
        3. Freezing the weights of the head module representing the classification layers.
        """
        assert isinstance(self.model, FedRepModel)
        self.fedrep_train_mode = FedRepTrainMode.REPRESENTATION
        self.model.unfreeze_base_module()
        self.model.freeze_head_module()

    def _prepare_train_head(self) -> None:
        """
        Handles the components switching needed to train the classification submodule as required by FedRep.

        This includes:

        1. Setting the training mode enum to know which optimizer should be stepping during training
        2. Freezing the base module, which represents the feature extractor.
        3. Unfreezing the weights of the head module representing the classification layers (if frozen).
        """
        assert isinstance(self.model, FedRepModel)
        self.fedrep_train_mode = FedRepTrainMode.HEAD
        self.model.unfreeze_head_module()
        self.model.freeze_base_module()

    def _prefix_loss_and_metrics_dictionaries(
        self, prefix: str, loss_dict: dict[str, float], metrics_dict: dict[str, Scalar]
    ) -> None:
        """
        This method is used to added the provided prefix string to the keys of both the loss_dict and the metrics_dict
        This function is used to separate the losses and metrics values obtained during local training of the head and
        feature extraction modules of FedRep, which occur separately and sequentially for the approach.

        Args:
            prefix (str): Prefix to be attached to all keys of the provided dictionaries.
            loss_dict (dict[str, float]): Dictionary of loss values obtained during training.
            metrics_dict (dict[str, Scalar]): Dictionary of metrics values measured during training.
        """
        for loss_key in list(loss_dict):
            loss_dict[f"{prefix}_{loss_key}"] = loss_dict.pop(loss_key)
        for metrics_key in list(metrics_dict):
            metrics_dict[f"{prefix}_{metrics_key}"] = metrics_dict.pop(metrics_key)

    def _extract_epochs_or_steps_specified(self, config: Config) -> EpochsAndStepsTuple:
        """
        Function parses the configuration specified and extracts the epochs or step based training values necessary
        to train a FedRep model. Note that we do not allow for mixed epoch and step based training. You must specify
        either epochs or steps for both the head and representation modules. The keys should be either
        ``{local_head_epochs, local_rep_epochs}`` or ``{local_head_steps, local_rep_steps}``.

        Args:
            config (Config): Configuration specifying all of the required parameters for training.

        Raises:
            ValueError: This function raises a value error in two scenarios. The first is when both steps and epochs
                have been specified for training the head and representation modules. The second is when epochs or
                steps values have not been specified for **BOTH** modules. This could also mean that the keys are
                wrong.

        Returns:
            EpochsAndStepsTuple: Returns a tuple of epochs and steps for which to train the head and representation
            modules. Only two of the four possible values will be defined, depending on whether we're doing
            epoch-based or step based training.
        """
        epochs_specified = ("local_head_epochs" in config) and ("local_rep_epochs" in config)
        steps_specified = ("local_head_steps" in config) and ("local_rep_steps" in config)
        if epochs_specified and not steps_specified:
            log(
                INFO,
                "Epochs for head and representation module specified. Proceeding with epoch-based training",
            )
            return (
                narrow_dict_type(config, "local_head_epochs", int),
                narrow_dict_type(config, "local_rep_epochs", int),
                None,
                None,
            )
        if steps_specified and not epochs_specified:
            log(
                INFO,
                "Steps for head and representation module specified. Proceeding with step-based training",
            )
            return (
                None,
                None,
                narrow_dict_type(config, "local_head_steps", int),
                narrow_dict_type(config, "local_rep_steps", int),
            )
        if epochs_specified and steps_specified:
            raise ValueError("Cannot specify both epochs and steps based training values in the config")
        raise ValueError(
            "Either configuration keys not properly present or a mix of steps and epochs based training was "
            "specified and is not admissible. Keys should be one of {local_head_epochs, local_rep_epochs} or "
            "{local_head_steps, local_rep_steps}"
        )

    def process_fed_rep_config(self, config: Config) -> tuple[EpochsAndStepsTuple, int, bool]:
        """
        Method to ensure the required keys are present in config and extracts values to be returned. We override this
        functionality from the ``BasicClient``, because ``FedRep`` has slightly different requirements. That is, one
        needs to specify a number of epochs or steps to do for BOTH the head module **AND** the representation module.

        Args:
            config (Config): The config from the server.

        Returns:
            tuple[int | None, int | None, int | None, int | None, int, bool]: Returns the ``local_epochs``,
            ``local_steps``, ``current_server_round`` and ``evaluate_after_fit``. Ensures only one of
            ``local_epochs`` and ``local_steps`` is defined in the config and sets the one that is not to None.

        Raises:
            ValueError: If the config contains both local_steps and local epochs or if ``local_steps``,
                ``local_epochs`` or ``current_server_round`` is of the wrong type (int).
        """
        current_server_round = narrow_dict_type(config, "current_server_round", int)
        steps_or_epochs_tuple = self._extract_epochs_or_steps_specified(config)

        try:
            evaluate_after_fit = narrow_dict_type(config, "evaluate_after_fit", bool)
        except ValueError:
            evaluate_after_fit = False

        # Either local epochs or local steps is none based on what key is passed in the config
        return steps_or_epochs_tuple, current_server_round, evaluate_after_fit

    def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys "representation" and "head,"
        respectively.
        """
        raise NotImplementedError

    def set_optimizer(self, config: Config) -> None:
        """
        FedRep requires an optimizer for the representations optimization and one for the model head. This function
        simply ensures that the optimizers setup by the user have the proper keys and that there are two optimizers.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and {"representation", "head"} == set(optimizers.keys()), (
            'Optimizer keys must be "representation" and "head" to use FedRep'
        )
        self.optimizers = optimizers

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        # Ensure that the model has the right type and setup the exchanger accordingly
        assert isinstance(self.model, SequentiallySplitExchangeBaseModel)
        return FixedLayerExchanger(self.model.layers_to_exchange())

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round) and performs training based on the passed config.
        For FedRep, this coordinates calling the right training functions based on the passed steps. We need to
        override the functionality of the ``BasicClient`` to allow for two distinct training passes of the model, as
        required by FedRep.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If the steps or epochs for the representation and head module training processes are are
                correctly specified.
        """
        round_start_time = datetime.datetime.now()
        (
            (local_head_epochs, local_rep_epochs, local_head_steps, local_rep_steps),
            current_server_round,
            evaluate_after_fit,
        ) = self.process_fed_rep_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config, fitting_round=True)

        self.update_before_train(current_server_round)

        fit_start_time = datetime.datetime.now()
        if local_head_epochs and local_rep_epochs:
            loss_dict, metrics = self.train_fedrep_by_epochs(local_head_epochs, local_rep_epochs, current_server_round)
        elif local_head_steps and local_rep_steps:
            loss_dict, metrics = self.train_fedrep_by_steps(local_head_steps, local_rep_steps, current_server_round)
        else:
            raise ValueError(
                "Local epochs or local steps have not been correctly specified. They have values "
                f"{local_head_epochs}, {local_rep_epochs}, {local_head_steps}, {local_rep_steps}"
            )
        fit_time = datetime.datetime.now() - fit_start_time

        # Check if we should run an evaluation with validation data after fit
        # (for example, this is used by FedDGGA)
        if self._should_evaluate_after_fit(evaluate_after_fit):
            validation_loss, validation_metrics = self.validate()
            metrics.update(validation_metrics)
            # We perform a pre-aggregation checkpoint if applicable
            self._maybe_checkpoint(validation_loss, validation_metrics, CheckpointMode.PRE_AGGREGATION)

        # Report data at end of round
        self.reports_manager.report(
            {
                "fit_metrics": metrics,
                "fit_losses": loss_dict,
                "round": current_server_round,
                "round_start": str(round_start_time),
                "fit_time_elapsed": str(fit_time),
            },
            current_server_round,
        )

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def train_fedrep_by_epochs(
        self, head_epochs: int, rep_epochs: int, current_round: int | None = None
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of epochs.

        Args:
            head_epochs (int): The number of epochs for local training of the head module.
            rep_epochs (int): The number of epochs for local training of the representation module
            current_round (int | None, optional): The current FL round. Defaults to None.

        Returns:
            tuple[dict[str, float], dict[str, Scalar]]: The loss and metrics dictionary from the local training.
            Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        # First we train the head module for head_epochs with the representations frozen in place
        self._prepare_train_head()
        log(INFO, f"Beginning FedRep Head Training Phase for {head_epochs} Epochs")
        loss_dict_head, metrics_dict_head = self.train_by_epochs(head_epochs, current_round)
        log(INFO, "Converting the loss and metrics dictionary keys for head training")
        # The loss and metrics coming from train_by_epochs are generically keyed, for example "backward." To avoid
        # clashing or being overwritten by the rep module training below, we prefix these keys.
        self._prefix_loss_and_metrics_dictionaries("head", loss_dict_head, metrics_dict_head)

        # Second we train the representation module for rep_epochs with the head module frozen in place
        self._prepare_train_representations()
        log(
            INFO,
            f"Beginning FedRep Representation Training Phase for {rep_epochs} Epochs",
        )
        loss_dict_rep, metrics_dict_rep = self.train_by_epochs(rep_epochs, current_round)
        log(INFO, "Converting the loss and metrics dictionary keys for Rep training")
        # The loss and metrics coming from train_by_epochs are generically keyed, for example "backward." To avoid
        # clashing or being overwritten by the head module training above, we prefix these keys.
        self._prefix_loss_and_metrics_dictionaries("rep", loss_dict_rep, metrics_dict_rep)
        log(INFO, "Merging the loss and training dictionaries")
        loss_dict_head.update(loss_dict_rep)
        metrics_dict_head.update(metrics_dict_rep)
        return loss_dict_head, metrics_dict_head

    def train_fedrep_by_steps(
        self, head_steps: int, rep_steps: int, current_round: int | None = None
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            head_steps (int): The number of steps to train locally for the head model.
            rep_steps (int): The number of steps to train locally for the representation model
            current_round (int | None, optional): What round of FL training we're currently on. Defaults to None.

        Returns:
            tuple[dict[str, float], dict[str, Scalar]]: The loss and metrics dictionary from the local training.
            Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        assert isinstance(self.model, FedRepModel)
        # First we train the head module for head_steps with the representations frozen in place
        self._prepare_train_head()
        log(INFO, f"Beginning FedRep Head Training Phase for {head_steps} Steps")
        loss_dict_head, metrics_dict_head = self.train_by_steps(head_steps, current_round)
        log(INFO, "Converting the loss and metrics dictionary keys for head training")
        # The loss and metrics coming from train_by_steps are generically keyed, for example "backward." To avoid
        # clashing or being overwritten by the rep module training below, we prefix these keys.
        self._prefix_loss_and_metrics_dictionaries("head", loss_dict_head, metrics_dict_head)

        # Second we train the representation module for rep_steps with the head module frozen in place
        self._prepare_train_representations()
        log(
            INFO,
            f"Beginning FedRep Representation Training Phase for {rep_steps} Steps",
        )
        loss_dict_rep, metrics_dict_rep = self.train_by_steps(rep_steps, current_round)
        log(INFO, "Converting the loss and metrics dictionary keys for Rep training")
        # The loss and metrics coming from train_by_steps are generically keyed, for example "backward." To avoid
        # clashing or being overwritten by the head module training above, we prefix these keys.
        self._prefix_loss_and_metrics_dictionaries("rep", loss_dict_rep, metrics_dict_rep)
        log(INFO, "Merging the loss and training dictionaries")
        loss_dict_head.update(loss_dict_rep)
        metrics_dict_head.update(metrics_dict_rep)
        return loss_dict_head, metrics_dict_head

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> tuple[TrainingLosses, TorchPredType]:
        """
        Mechanics of training loop follow the FedRep paper: https://arxiv.org/pdf/2102.07078.pdf. In order to reuse
        the ``train_step`` functionality, we switch between the appropriate optimizers depending on the
        clients training mode (HEAD vs. REPRESENTATION).

        Args:
            input (TorchInputType): input tensor to be run through the model. Here, ``TorchInputType`` is simply an
                alias for the union of ``torch.Tensor`` and ``dict[str, torch.Tensor]``.
            target (torch.Tensor): target tensor to be used to compute a loss given the model's outputs.

        Returns:
            tuple[TrainingLosses, dict[str, torch.Tensor]]: The losses object from the train step along with
            a dictionary of any predictions produced by the model.
        """
        # Clear gradients from the optimizers if they exits. We do both regardless of the client mode.
        self.optimizers["representation"].zero_grad()
        self.optimizers["head"].zero_grad()

        # Perform forward pass on the full model
        preds, features = self.predict(input)
        target = self.transform_target(target)  # Apply transformation (Defaults to identity)

        # Compute all relevant losses
        losses = self.compute_training_loss(preds, features, target)
        losses.backward["backward"].backward()

        if self.fedrep_train_mode == FedRepTrainMode.HEAD:
            self.optimizers["head"].step()
        elif self.fedrep_train_mode == FedRepTrainMode.REPRESENTATION:
            self.optimizers["representation"].step()
        else:
            raise ValueError("Training Mode in an invalid state")

        # Return dictionary of predictions where key is used to name respective MetricMeters
        return losses, preds
