from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.adaptive_drift_constraint_client import AdaptiveDriftConstraintClient
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchPredType, TorchTargetType


class MrMtlClient(AdaptiveDriftConstraintClient):
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
        This client implements the MR-MTL algorithm from MR-MTL: On Privacy and Personalization in Cross-Silo
        Federated Learning. The idea is that we want to train personalized versions of the global model for each
        client. However, instead of using a separate solver for the global model, as in Ditto, we update the initial
        global model with aggregated local models on the server-side and use those weights to also constrain the
        training of a local model. The constraint for this local model is identical to the FedProx loss. The key
        difference is that the local model is never replaced with aggregated weights. It is always local.

        NOTE: lambda, the drift loss weight, is initially set and potentially adapted by the server akin to the
        heuristic suggested in the original FedProx paper. Adaptation is optional and can be disabled in the
        corresponding strategy used by the server

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
        # NOTE: The initial global model is used to house the aggregate weight updates at the beginning of a round,
        # because in MR-MTL, the local models are not updated with these aggregates.
        self.initial_global_model: nn.Module
        self.initial_global_tensors: List[torch.Tensor]

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        # Need to setup the init global model here as well. It should be the same architecture as the model so
        # we reuse the get_model call. We explicitly send the model to the desired device. This is idempotent.
        self.initial_global_model = self.get_model(config).to(self.device)
        # The rest of the setup is the same
        super().setup_client(config)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        The parameters being passed are to be routed to the initial global model to be used in a penalty term in
        training the local model. Despite the usual FL setup, we actually never pass the aggregated model to the
        LOCAL model. Instead, we use the aggregated model to form the MR-MTL penalty term.

        NOTE; In MR-MTL, unlike Ditto, the local model weights are not synced across clients to the initial global
        model, even in the FIRST ROUND.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model. It will also contain a penalty weight from the server at each round (possibly adapted)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is a fitting
                round or an evaluation round. Not used here.
        """
        # Make sure that the proper components exist.
        assert self.initial_global_model is not None and self.parameter_exchanger is not None

        # Route the parameters to the GLOBAL model only in MR-MTL
        log(INFO, "Setting the global model weights")
        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Lambda weight received from the server: {self.drift_penalty_weight}")

        self.parameter_exchanger.pull_parameters(server_model_state, self.initial_global_model, config)

    def update_before_train(self, current_server_round: int) -> None:
        assert self.initial_global_model is not None
        # Freeze the initial weights of the INITIAL GLOBAL MODEL. These are used to form the MR-MTL
        # update penalty term.
        for param in self.initial_global_model.parameters():
            param.requires_grad = False
        self.initial_global_model.eval()

        # Saving the initial GLOBAL MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the MR-MTL local update penalty term.
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.initial_global_model.parameters()
        ]

        return super().update_before_train(current_server_round)

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the modes and ground truth data. We add to vanilla loss
        function by including Mean Regularized (MR) penalty loss which is the l2 inner product between the
        initial global model weights and weights of the current model.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses indexed by
                name. Additional losses includes each loss component of the total loss.
        """
        # Check that the initial global model isn't in training mode and that the local model is in training mode
        assert not self.initial_global_model.training and self.model.training
        # Use the rest of the training loss computation from the AdaptiveDriftConstraintClient parent
        return super().compute_training_loss(preds, features, target)

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The validation loss and a dictionary of metrics from validation.
        """
        # ensure that the initial global model is in eval mode
        assert not self.initial_global_model.training
        return super().validate()
