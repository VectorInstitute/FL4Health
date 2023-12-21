from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from fl4health.utils.losses import Losses, LossMeterType
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
        checkpointer: Optional[TorchCheckpointer] = None,
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

    def get_proximal_loss(self) -> torch.Tensor:
        assert self.initial_tensors is not None
        # Using state dictionary to ensure the same ordering as exchange
        model_weights = [layer_weights for layer_weights in self.model.parameters()]
        assert len(self.initial_tensors) == len(model_weights)

        layer_inner_products: List[torch.Tensor] = [
            torch.pow(torch.linalg.norm(initial_layer_weights - iteration_layer_weights), 2.0)
            for initial_layer_weights, iteration_layer_weights in zip(self.initial_tensors, model_weights)
        ]

        # network l2 inner product tensor
        # NOTE: Scaling by 1/2 is for consistency with the original fedprox paper.
        return (self.proximal_weight / 2.0) * torch.stack(layer_inner_products).sum()

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

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with proximal weight. They are
        unpacked for the clients to use in training. If it's the first time the model is being initialized, we assume
        the full model is being  initialized and use the FullParameterExchanger() to set all model weights
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the proximal weight to be applied during training.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.proximal_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Proximal weight received from the server: {self.proximal_weight}")

        super().set_parameters(server_model_state, config)

        # Saving the initial weights and detaching them so that we don't compute gradients with respect to the
        # tensors. These are used to form the FedProx loss.
        self.initial_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.model.parameters()
        ]

    def compute_loss(
        self, preds: Dict[str, torch.Tensor], features: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> Losses:
        """
        Computes loss given predictions of the model and ground truth data. Adds to objective by including
        proximal loss which is the l2 norm between the initial and final weights of local training.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            Losses: Object containing checkpoint loss, backward loss and additional losses indexed by name.
            Additional losses includes proximal loss.
        """
        loss = self.criterion(preds["prediction"], target)
        proximal_loss = self.get_proximal_loss()
        total_loss = loss + proximal_loss
        losses = Losses(checkpoint=loss, backward=total_loss, additional_losses={"proximal_loss": proximal_loss})
        return losses

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return ParameterExchangerWithPacking(ParameterPackerFedProx())

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        """
        Called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionary.
        """
        # Store current loss which is the vanilla loss without the proximal term added in
        self.current_loss = loss_dict["checkpoint"]
