from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.client_side_module import ClientSideCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class MrMtlClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientSideCheckpointModule] = None,
        lam: float = 1.0,
    ) -> None:
        """
        This client implements the MR-MTL algorithm from MR-MTL: On Privacy and Personalization in Cross-Silo
        Federated Learning. The idea is that we want to train personalized versions of the global model for each
        client. However, instead of using a separate solver for the global model, as in Ditto, we update the initial
        global model with aggregated local models on the server-side and use those weights to also constrain the
        training of a local model. The constraint for this local model is identical to the FedProx loss.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
            lam (float, optional): weight applied to the MR-MTL drift loss. Defaults to 1.0.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.lam = lam
        self.init_global_model: nn.Module

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        # Need to setup the init global model here as well. It should be the same architecture as the model so
        # we reuse the get_model call. We explicitly send the model to the desired device. This is idempotent.
        self.init_global_model = self.get_model(config).to(self.device)
        # The rest of the setup is the same
        super().setup_client(config)

    def get_mr_drift_loss(self) -> torch.Tensor:
        """
        Compute the L2 inner product between the initial global weights for the round and the current local model
            weights. This loss function is added to the loss function for the local model when back propagating.

        Returns:
            torch.Tensor: Returns the L2 inner product between the initial global weights of the round and the current
                local model weights.
        """
        assert self.init_global_model is not None and self.model is not None and self.lam is not None
        # Using parameters to ensure the same ordering as exchange
        local_model_weights = [layer_weights for layer_weights in self.model.parameters()]
        # Detach the weights to ensure we don't compute gradients with respect to the tensors
        initial_global_weights = [layer_weights.detach() for layer_weights in self.init_global_model.parameters()]

        assert len(initial_global_weights) == len(local_model_weights)
        assert len(initial_global_weights) > 0

        layer_inner_products: List[torch.Tensor] = [
            torch.pow(torch.linalg.norm(initial_layer_weights - iteration_layer_weights), 2.0)
            for initial_layer_weights, iteration_layer_weights in zip(initial_global_weights, local_model_weights)
        ]

        # network l2 inner product tensor weighted by lambda
        return (self.lam / 2.0) * torch.stack(layer_inner_products).sum()

    def get_parameters(self, config: Config) -> NDArrays:
        """
        For MR-MTL, we transfer the LOCAL model weights to the server to be aggregated and set as INITIAL GLOBAL model
        weights on client side.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: LOCAL model weights to be sent to the server for aggregation
        """
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        The parameters being pass are to be routed to the initial global model to be used in a penalty term in
        training the local model. Despite the usual FL setup, we actually never pass the aggregated model to the
        LOCAL model. Instead, we use the aggregated model to form the MR-MTL penalty term.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model (global model for all but the first step of MR-MTL)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning
                round is a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round.
        """
        # Make sure that the proper components exist.
        assert self.init_global_model is not None and self.model is not None
        assert self.parameter_exchanger is not None and isinstance(self.parameter_exchanger, FullParameterExchanger)

        # Route the parameters to the GLOBAL model in MR-MTL
        assert self.parameter_exchanger is not None
        log(INFO, "Setting the global model weights")
        self.parameter_exchanger.pull_parameters(parameters, self.init_global_model, config)

    def update_before_train(self, current_server_round: int) -> None:
        assert isinstance(self.init_global_model, nn.Module)
        # Freeze the initial weights INIT GLOBAL MODEL. These are used to form the MR-MTL
        # update penalty term.
        for param in self.init_global_model.parameters():
            param.requires_grad = False
        self.init_global_model.eval()

        return super().update_before_train(current_server_round)

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the modes and ground truth data. We add to vanilla loss
        function by including Mean Regularized (MR) penalty loss which is the l2 inner product between the
        initial global model weights and weights of the current model.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and
                additional losses indexed by name. Additional losses includes each loss component of the total loss.
        """
        # Check that both models are in training mode
        assert not self.init_global_model.training and self.model.training

        total_loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        if additional_losses is None:
            additional_losses = {}

        # Compute mr-mtl drift loss
        mr_local_loss = self.get_mr_drift_loss()
        additional_losses["mr_loss"] = mr_local_loss

        return TrainingLosses(backward=total_loss + mr_local_loss, additional_losses=additional_losses)

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The validation loss and a dictionary of metrics from validation.
        """
        # Set the global model to evaluate mode
        self.init_global_model.eval()
        return super().validate()
