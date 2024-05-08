from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class FedProxClient(BasicClient):
    """
    This client implements the FedProx algorithm from Federated Optimization in Heterogeneous Networks. The idea is
    fairly straightforward. The local loss for each client is augmented with a norm on the difference between the
    local client weights during training (w) and the initial globally shared weights (w^t).
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.initial_tensors: List[torch.Tensor]
        self.parameter_exchanger: ParameterExchangerWithPacking
        self.proximal_weight: float
        self.current_loss: float
        self.proximal_loss_function = WeightDriftLoss(self.device)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Packs the parameters and training loss into a single NDArrays to be sent to the server for aggregation
        """
        assert self.model is not None and self.parameter_exchanger is not None and self.current_loss is not None

        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and training loss sent to server for aggregation
        # Training loss sent because server will decide to increase or decrease the proximal weight
        # Therefore it can only be computed locally
        packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.current_loss)
        return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with proximal weight. They are
        unpacked for the clients to use in training. In the first fitting round, we assume the full model is being
        initialized and use the FullParameterExchanger() to set all model weights.
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the proximal weight to be applied during training.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning
                round is a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.proximal_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Proximal weight received from the server: {self.proximal_weight}")

        super().set_parameters(server_model_state, config, fitting_round)

    def update_before_train(self, current_server_round: int) -> None:
        # Saving the initial weights and detaching them so that we don't compute gradients with respect to the
        # tensors. These are used to form the FedProx loss.
        self.initial_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.model.parameters()
        ]

        return super().update_before_train(current_server_round)

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions of the model and ground truth data. Adds to objective by including
        proximal loss which is the l2 norm between the initial and final weights of local training.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and
                additional losses indexed by name. Additional losses includes proximal loss.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        if additional_losses is None:
            additional_losses = {}

        proximal_loss = self.proximal_loss_function(self.model, self.initial_tensors, self.proximal_weight)
        additional_losses["proximal_loss"] = proximal_loss
        # adding the vanilla loss to the additional losses to be used by update_after_train
        additional_losses["loss"] = loss

        return TrainingLosses(backward=loss + proximal_loss, additional_losses=additional_losses)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return ParameterExchangerWithPacking(ParameterPackerFedProx())

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        """
        Called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionary.
        """
        # Store current loss which is the vanilla loss without the proximal term added in
        self.current_loss = loss_dict["loss"]
