from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import torch
from flwr.common import NDArrays
from flwr.common.logger import log
from flwr.common.typing import Config
from numpy import linalg

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithClippingBit
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType


class NumpyClippingClient(BasicClient):
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
        Client that clips updates being sent to the server where noise is added. Used to obtain Client Level
        Differential Privacy in FL setting.

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
        self.parameter_exchanger: FullParameterExchangerWithPacking[float]
        self.clipping_bound: float | None = None
        self.adaptive_clipping: bool | None = None

    def calculate_parameters_norm(self, parameters: NDArrays) -> float:
        """
        Given a set of parameters, compute the l2-norm of the parameters. This is a matrix norm: squared sum of all of
        the weights.

        Args:
            parameters (NDArrays): Tensor to measure with the norm

        Returns:
            (float): Squared sum of all values in the NDArrays
        """
        layer_inner_products = [pow(linalg.norm(layer_weights), 2) for layer_weights in parameters]
        # network Frobenius norm
        return pow(sum(layer_inner_products), 0.5)

    def clip_parameters(self, parameters: NDArrays) -> tuple[NDArrays, float]:
        """
        Performs "flat clipping" on the parameters as follows.

        \\[\\text{parameters} \\cdot \\min \\left(1, \\frac{C}{\\Vert \\text{parameters} \\Vert_2} \\right)\\]

        Args:
            parameters (NDArrays): Parameters to clip

        Returns:
            (tuple[NDArrays, float]): Clipped parameters and the associated clipping bit indicating whether the norm
                was below ``self.clipping_bound``. If ``self.adaptive_clipping`` is false, this bit is always 0.0
        """
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

    def compute_weight_update_and_clip(self, parameters: NDArrays) -> tuple[NDArrays, float]:
        """
        Compute the weight delta (i.e. new weights - old weights) and clip according to ``self.clipping_bound``.

        Args:
            parameters (NDArrays): Updated parameters to compute the delta from and clip thereafter.

        Returns:
            (tuple[NDArrays, float]): Clipped weighted updates (weight deltas) and the associated clipping bit.
        """
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
        This function performs clipping through ``compute_weight_update_and_clip`` and stores the clipping bit
        as the last entry in the NDArrays.
        """
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        if not self.initialized or current_server_round == 0:
            # If we haven't initialized the client we are being asked to return model parameters. So we send them all
            return self.setup_client_and_return_all_model_parameters(config)

        assert self.model is not None and self.parameter_exchanger is not None
        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)
        clipped_weight_update, clipping_bit = self.compute_weight_update_and_clip(model_weights)
        return self.parameter_exchanger.pack_parameters(clipped_weight_update, clipping_bit)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        This function assumes that the parameters being passed contain model parameters followed by the last entry
        of the list being the new clipping bound. They are unpacked for the clients to use in training. If it is
        called in the first fitting round, we assume the full model is being initialized and use the
        ``FullParameterExchanger()`` to set all model weights.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the clipping bound.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round
                is a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling
                parameters.
                A full parameter exchanger is used if the current federated learning round is the very
                first fitting round.
        """
        assert self.model is not None and self.parameter_exchanger is not None
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        server_model_parameters, clipping_bound = self.parameter_exchanger.unpack_parameters(parameters)
        self.clipping_bound = clipping_bound
        current_server_round = narrow_dict_type(config, "current_server_round", int)

        if current_server_round == 1 and fitting_round:
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
        return FullParameterExchangerWithPacking(ParameterPackerWithClippingBit())

    def setup_client(self, config: Config) -> None:
        self.adaptive_clipping = narrow_dict_type(config, "adaptive_clipping", bool)
        super().setup_client(config)
