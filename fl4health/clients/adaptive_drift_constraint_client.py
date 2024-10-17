from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchPredType, TorchTargetType


class AdaptiveDriftConstraintClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
    ) -> None:
        """
        This client serves as a base for FL methods implementing an auxiliary loss penalty with a weight coefficient
        that might be adapted via loss trajectories on the server-side. An example of such a method is FedProx, which
        uses an auxiliary loss penalizing weight drift with a coefficient mu. This client is a simple extension of the
        BasicClient that packs the self.loss_for_adaptation for exchange with the server and expects to receive an
        updated (or constant if non-adaptive) parameter for the loss weight. In many cases, such as FedProx, the
        loss_for_adaptation being packaged is the criterion loss (i.e. loss without the penalty)

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
            progress_bar (bool): Whether or not to display a progress bar during client training and validation.
                Uses tqdm. Defaults to False
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            reporters=reporters,
            progress_bar=progress_bar,
        )
        # These are the tensors that will be used to compute the penalty loss
        self.drift_penalty_tensors: List[torch.Tensor]
        # Exchanger with packing to be able to exchange the weights and auxiliary information with the server for
        # adaptation
        self.parameter_exchanger: FullParameterExchangerWithPacking
        # Weight on the penalty loss to be used in backprop. This is what might be adapted via server calculations
        self.drift_penalty_weight: float
        # This is the loss value to be sent back to the server on which adaptation decisions will be made.
        self.loss_for_adaptation: float
        # Function to compute the penalty loss.
        self.penalty_loss_function = WeightDriftLoss(self.device)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Packs the parameters and training loss into a single NDArrays to be sent to the server for aggregation. If
        the client has not been initialized, this means the server is requesting parameters for initialization and
        just the model parameters are sent. When using the FedAvgWithAdaptiveConstraint strategy, this should not
        happen, as that strategy requires server-side initialization parameters. However, other strategies may handle
        this case.

        Args:
            config (Config): Configurations to allow for customization of this functions behavior

        Returns:
            NDArrays: Parameters and training loss packed together into a list of numpy arrays to be sent to the server
        """
        if not self.initialized:
            log(INFO, "Setting up client and providing full model parameters to the server for initialization")

            # If initialized is False, the server is requesting model parameters from which to initialize all other
            # clients. As such get_parameters is being called before fit or evaluate, so we must call
            # setup_client first.
            self.setup_client(config)

            # Need all parameters even if normally exchanging partial
            return FullParameterExchanger().push_parameters(self.model, config=config)
        else:

            # Make sure the proper components are there
            assert (
                self.model is not None
                and self.parameter_exchanger is not None
                and self.loss_for_adaptation is not None
            )
            model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

            # Weights and training loss sent to server for aggregation. Training loss is sent because server will
            # decide to increase or decrease the penalty weight, if adaptivity is turned on.
            packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.loss_for_adaptation)
            return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with a penalty weight. They are
        unpacked for the clients to use in training. In the first fitting round, we assume the full model is being
        initialized and use the FullParameterExchanger() to set all model weights.
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the penalty weight to be applied during training.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is a fitting
                round or an evaluation round. This is used to help determine which parameter exchange should be used
                for pulling parameters. A full parameter exchanger is always used if the current federated learning
                round is the very first fitting round.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Penalty weight received from the server: {self.drift_penalty_weight}")

        super().set_parameters(server_model_state, config, fitting_round)

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions of the model and ground truth data. Adds to objective by including
        penalty loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. All predictions included in
                dictionary will be used to compute metrics.
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses indexed
                by name. Additional losses includes penalty loss.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        if additional_losses is None:
            additional_losses = {}

        additional_losses["loss"] = loss.clone()
        # adding the vanilla loss to the additional losses to be used by update_after_train for potential adaptation
        additional_losses["loss_for_adaptation"] = loss.clone()

        # Compute the drift penalty loss and store it in the additional losses dictionary.
        penalty_loss = self.compute_penalty_loss()
        additional_losses["penalty_loss"] = penalty_loss.clone()

        return TrainingLosses(backward=loss + penalty_loss, additional_losses=additional_losses)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Setting up the parameter exchanger to include the appropriate packing functionality.
        By default we assume that we're exchanging all parameters. Can be overridden for other behavior
        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            ParameterExchanger: Exchanger that can handle packing/unpacking auxiliary server information.
        """

        return FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float], config: Config) -> None:
        """
        Called after training with the number of local_steps performed over the FL round and the corresponding loss
        dictionary. We use this to store the training loss that we want to use to adapt the penalty weight parameter
        on the server side.

        Args:
            local_steps (int): The number of steps so far in the round in the local training.
            loss_dict (Dict[str, float]): A dictionary of losses from local training.
            config (Config): The config from the server
        """
        assert "loss_for_adaptation" in loss_dict
        # Store current loss which is the vanilla loss without the penalty term added in
        self.loss_for_adaptation = loss_dict["loss_for_adaptation"]
        super().update_after_train(local_steps, loss_dict, config)

    def compute_penalty_loss(self) -> torch.Tensor:
        """
        Computes the drift loss for the client model and drift tensors

        Returns:
            torch.Tensor: Computed penalty loss tensor
        """
        # Penalty tensors must have been set for these clients.
        assert self.drift_penalty_tensors is not None

        return self.penalty_loss_function(self.model, self.drift_penalty_tensors, self.drift_penalty_weight)
