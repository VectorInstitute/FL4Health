from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx
from fl4health.utils.metrics import MeterType, Metric


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
        meter_type: MeterType = MeterType.AVERAGE,
        use_wandb_reporter: bool = False,
        checkpointer: Optional[TorchCheckpointer] = None,
        proximal_weight: float = 0.1,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            meter_type=meter_type,
            use_wandb_reporter=use_wandb_reporter,
            checkpointer=checkpointer,
        )
        self.proximal_weight = proximal_weight
        self.initial_tensors: List[torch.Tensor]
        self.parameter_exchanger: ParameterExchangerWithPacking

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
        assert self.model is not None and self.parameter_exchanger is not None

        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and training loss sent to server for aggregation
        # Training loss sent because server will decide to increase or decrease the proximal weight
        # Therefore it can only be computed locally
        assert self.current_loss is not None
        packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.current_loss)
        return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with
        proximal weight. They are unpacked for the clients to use in training.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.proximal_weight = self.parameter_exchanger.unpack_parameters(parameters)

        self.server_model_state = server_model_state
        self.parameter_exchanger.pull_parameters(server_model_state, self.model, config)

        """
        Assumes that the parameters being passed contain model parameters concatenated with
        proximal weight. They are unpacked for the clients to use in training.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.proximal_weight = self.parameter_exchanger.unpack_parameters(parameters)

        self.server_model_state = server_model_state
        self.parameter_exchanger.pull_parameters(server_model_state, self.model, config)

        # Saving the initial weights and detaching them so that we don't compute gradients with respect to the
        # tensors. These are used to form the FedProx loss.
        self.initial_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.model.parameters()
        ]

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        local_epochs, local_steps, current_server_round = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)

        if local_epochs is not None:
            losses, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = self.num_train_samples * local_epochs  # total steps over training round
        else:
            assert isinstance(local_steps, int)
            losses, metrics = self.train_by_steps(local_steps, current_server_round)

        # Store current losses (Used by FedProx Client)
        self.current_loss = losses["loss"]

        # Update model after train round (Used by Scaffold and DP-Scaffold Client)
        self.update_after_train(local_steps)

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss = self.criterion(preds, target)
        proximal_loss = self.get_proximal_loss()
        total_loss = loss + proximal_loss
        return loss, {"proximal_loss": proximal_loss, "total_loss": total_loss}

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        return ParameterExchangerWithPacking(ParameterPackerFedProx())
