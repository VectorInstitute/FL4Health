import copy
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays

from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.layer_exchanger import NormDriftParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric


class DynamicWeightExchangeClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
    ) -> None:
        """
        Dynamic weight exchange client used to exchange a dynamic subset of layers per client.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
        )
        # Initial model parameters to be used in calculating weight shifts during training
        self.initial_model: nn.Module
        # Parameter exchanger to be used in server-client exchange of dynamic layers.
        self.parameter_exchanger: NormDriftParameterExchanger

    def setup_client(self, config: Config) -> None:
        """
        Setup the components of the client neccessary for client side training and parameter exchange. Mostly handled
        by a call to the basic client flow, but also sets up the initial model to facilitate storage of initial
        parameters during training

        Args:
            config (Config): Configuration used to setup the client properly
        """
        super().setup_client(config)
        self.initial_model = copy.deepcopy(self.model).to(self.device)

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        This method configures and instantiates a NormDriftParameterExchanger to be used in dynamic weight exchange.

        Args:
            config (Config): Configuration used to setup the weight exchanger properties for dynamic exchange

        Returns:
            ParameterExchanger: This exchanger handles the exchange orchestration between clients and server during
                federated training
        """
        normalize = self.narrow_config_type(config, "normalize", bool)
        filter_by_percentage = self.narrow_config_type(config, "filter_by_percentage", bool)
        norm_threshold = self.narrow_config_type(config, "norm_threshold", float)
        exchange_percentage = self.narrow_config_type(config, "exchange_percentage", float)
        parameter_exchanger = NormDriftParameterExchanger(
            norm_threshold, exchange_percentage, normalize, filter_by_percentage
        )
        return parameter_exchanger

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

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Sets the local model parameters transfered from the server using a parameter exchanger to coordinate how
        parameters are set. If it's the first time the model is being initialized, we assume the full model is
        being initialized and the weights sent correspond to the complete set of weights. Thus we use the
        FullParameterExchanger() to set all model weights.

        Subsequently, this approach uses the threshold parameter exchanger to handle exchanging a dynamic subset of
        model layer weights.

        Args:
            parameters (NDArrays): parameters is the set of weights and their corresponding model component names,
                corresponding to the state dict names. These are woven together in the NDArrays object. These are
                unwound properly by the parameter exchanger
            config (Config): configuration if required to control parameter exchange.
        """
        if not self.model_weights_initialized:
            self.initialize_all_model_weights(parameters, config)
        else:
            self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        # stores the values of the new model parameters at the beginning of each client training round.
        self.initial_model.load_state_dict(self.model.state_dict(), strict=True)
