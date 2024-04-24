import copy
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class PartialWeightExchangeClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
        store_initial_model: bool = False,
    ) -> None:
        """
        Client that only exchanges a subset of its parameters with the server in each communication round.

        The strategy for selecting which parameters to exchange is determined by self.parameter_exchanger,
        which must be a subclass of PartialParameterExchanger.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            store_initial_model (bool): Indicates whether the client should store a copy of the model weights
                at the beginning of each training round. The model copy might be required to select the subset
                of model parameters to be exchanged with the server, depending on the selection criterion used.
                Defaults to False.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            metrics_reporter=metrics_reporter,
        )
        # Initial model parameters to be used in selecting parameters to be exchanged during training.
        self.initial_model: Optional[nn.Module]
        # Parameter exchanger to be used in server-client exchange of dynamic layers.
        self.parameter_exchanger: PartialParameterExchanger
        self.store_initial_model = store_initial_model

    def setup_client(self, config: Config) -> None:
        """
        Setup the components of the client necessary for client side training and parameter exchange. Mostly handled
        by a call to the basic client flow, but also sets up the initial model to facilitate storage of initial
        parameters during training

        Args:
            config (Config): Configuration used to setup the client properly
        """
        super().setup_client(config)
        if self.store_initial_model:
            self.initial_model = copy.deepcopy(self.model).to(self.device)
        else:
            self.initial_model = None

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        This method configures and instantiates a PartialParameterExchanger and should be
        implemented by the user since there are various strategies to select
        parameters to be exchanged.

        Args:
            config (Config): Configuration used to setup the weight exchanger properties for dynamic exchange

        Returns:
            ParameterExchanger: This exchanger handles the exchange orchestration between clients and server during
                federated training
        """
        raise NotImplementedError

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Determines which weights are sent back to the server for aggregation. This uses a parameter exchanger to
        determine parameters sent. Note that this overrides the basic client get_parameters function to send the
        initial model so that starting weights may be extracted and compared to current weights after local
        training

        Args:
            config (Config): configuration used to setup the exchange

        Returns:
            NDArrays: The list of weights to be sent to the server from the client
        """
        assert self.model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.model, self.initial_model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        parameters are set.

        In the first fitting round, we assume the full model is being
        initialized and use the FullParameterExchanger() to set all model weights.

        In other times, this approach uses a partial weight exchanger to set model weights.

        Args:
            parameters (NDArrays): parameters is the set of weights and their corresponding model component names,
                corresponding to the state dict names. These are woven together in the NDArrays object. These are
                unwound properly by the parameter exchanger
            config (Config): configuration if required to control parameter exchange.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is
                a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling
                parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round. Otherwise, use a PartialParameterExchanger.
        """
        super().set_parameters(parameters, config, fitting_round)
        if self.store_initial_model:
            assert self.initial_model is not None
            # Stores the values of the new model parameters at the beginning of each client training round.
            self.initial_model.load_state_dict(self.model.state_dict(), strict=True)
