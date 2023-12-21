from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from flwr.common import NDArrays
from flwr.common.logger import log
from flwr.common.typing import Config
from numpy import linalg

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithClippingBit
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class NumpyClippingClient(BasicClient):
    """
    Client that clips updates being sent to the server where noise is added.
    Used to obtain Client Level Differenital Privacy in FL setting.
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
        self.parameter_exchanger: ParameterExchangerWithPacking[float]
        self.clipping_bound: Optional[float] = None
        self.adaptive_clipping: Optional[bool] = None

    def calculate_parameters_norm(self, parameters: NDArrays) -> float:
        layer_inner_products = [pow(linalg.norm(layer_weights), 2) for layer_weights in parameters]
        # network froebenius norm
        return pow(sum(layer_inner_products), 0.5)

    def clip_parameters(self, parameters: NDArrays) -> Tuple[NDArrays, float]:
        assert self.clipping_bound is not None
        assert self.adaptive_clipping is not None
        # performs flat clipping (i.e. parameters * min(1, C/||parameters||_2))
        network_frobenius_norm = self.calculate_parameters_norm(parameters)
        log(INFO, f"Update norm: {network_frobenius_norm}, Clipping Bound: {self.clipping_bound}")
        if network_frobenius_norm <= self.clipping_bound:
            # if we're not adaptively clipping then don't send true clipping bit info as this would potentially leak
            # information
            clipping_bit = 1.0 if self.adaptive_clipping else 0.0
            return parameters, clipping_bit
        clip_scalar = min(1.0, self.clipping_bound / network_frobenius_norm)
        # parameters and clipping bit
        return [layer_weights * clip_scalar for layer_weights in parameters], 0.0

    def compute_weight_update_and_clip(self, parameters: NDArrays) -> Tuple[NDArrays, float]:
        assert self.initial_weights is not None
        assert len(parameters) == len(self.initial_weights)
        weight_update: NDArrays = [
            new_layer_weights - old_layer_weights
            for old_layer_weights, new_layer_weights in zip(self.initial_weights, parameters)
        ]
        # return clipped parameters and clipping bit
        return self.clip_parameters(weight_update)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        This function performs clipping through compute_weight_update_and_clip and stores the clipping bit
        as the last entry in the NDArrays
        """
        assert self.model is not None and self.parameter_exchanger is not None
        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)
        clipped_weight_update, clipping_bit = self.compute_weight_update_and_clip(model_weights)
        return self.parameter_exchanger.pack_parameters(clipped_weight_update, clipping_bit)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        This function assumes that the parameters being passed contain model parameters followed by the last entry
        of the list being the new clipping bound. They are unpacked for the clients to use in training. If it's the
        first time the model is being initialized, we assume the full model is being initialized and use the
        FullParameterExchanger() to set all model weights

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the clipping bound.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        assert self.model is not None and self.parameter_exchanger is not None
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        server_model_parameters, clipping_bound = self.parameter_exchanger.unpack_parameters(parameters)
        self.clipping_bound = clipping_bound

        if not self.model_weights_initialized:
            # Initialize all model weights as this is the first time things have been set
            self.initialize_all_model_weights(server_model_parameters, config)
            # Extract only the initial weights that we care about clipping and exchanging
            self.initial_weights = self.parameter_exchanger.push_parameters(self.model, config=config)
        else:
            # Store the starting parameters without clipping bound before client optimization steps
            self.initial_weights = server_model_parameters
            # Inject the server model parameters into the client model
            self.parameter_exchanger.pull_parameters(server_model_parameters, self.model, config)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithClippingBit())
        return parameter_exchanger

    def setup_client(self, config: Config) -> None:
        self.adaptive_clipping = self.narrow_config_type(config, "adaptive_clipping", bool)
        super().setup_client(config)
