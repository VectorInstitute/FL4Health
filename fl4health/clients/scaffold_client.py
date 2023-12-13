import copy
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.clients.instance_level_privacy_client import InstanceLevelPrivacyClient
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.utils.losses import Losses, LossMeterType
from fl4health.utils.metrics import Metric

ScaffoldTrainStepOutput = Tuple[torch.Tensor, torch.Tensor]


class ScaffoldClient(BasicClient):
    """
    Federated Learning Client for Scaffold strategy.

    Implementation based on https://arxiv.org/pdf/1910.06378.pdf.
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
        self.learning_rate: float  # eta_l in paper
        self.client_control_variates: Optional[NDArrays] = None  # c_i in paper
        self.client_control_variates_updates: Optional[NDArrays] = None  # delta_c_i in paper
        self.server_control_variates: Optional[NDArrays] = None  # c in paper
        self.optimizer: torch.optim.SGD  # Scaffold require vanilla SGD as optimizer
        self.server_model_weights: Optional[NDArrays] = None  # x in paper
        self.parameter_exchanger: ParameterExchangerWithPacking[NDArrays]

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Packs the parameters and control variartes into a single NDArrays to be sent to the server for aggregation
        """
        assert self.model is not None and self.parameter_exchanger is not None

        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and control variates updates sent to server for aggregation
        # Control variates updates sent because only client has access to previous client control variate
        # Therefore it can only be computed locally
        assert self.client_control_variates_updates is not None
        packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.client_control_variates_updates)
        return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with server control variates.
        They are unpacked for the clients to use in training. If it's the first time the model is being initialized,
        we assume the full model is being initialized and use the FullParameterExchanger() to set all model weights
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the server control variates (initial or after aggregation)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, server_control_variates = self.parameter_exchanger.unpack_parameters(parameters)
        self.server_control_variates = server_control_variates

        super().set_parameters(server_model_state, config)

        # Note that we are restricting to weights that require a gradient here because they are used to compute
        # control variates
        self.server_model_weights = [
            model_params.cpu().detach().clone().numpy()
            for model_params in self.model.parameters()
            if model_params.requires_grad
        ]

        # If client control variates do not exist, initialize them to be the same as the server control variates.
        # Server variates default to be 0, but as stated in the paper the control variates should be the uniform
        # average of the client variates. So if server_control_variates are non-zero, this ensures that average
        # still holds.
        if self.client_control_variates is None:
            self.client_control_variates = copy.deepcopy(self.server_control_variates)

    def update_control_variates(self, local_steps: int) -> None:
        """
        Updates local control variates along with the corresponding updates
        according to the option 2 in Equation 4 in https://arxiv.org/pdf/1910.06378.pdf
        To be called after weights of local model have been updated.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None
        assert self.server_model_weights is not None
        assert self.learning_rate is not None

        # y_i
        client_model_weights = [
            val.cpu().detach().clone().numpy() for val in self.model.parameters() if val.requires_grad
        ]

        # (x - y_i)
        delta_model_weights = self.compute_parameters_delta(self.server_model_weights, client_model_weights)

        # (c_i - c)
        delta_control_variates = self.compute_parameters_delta(
            self.client_control_variates, self.server_control_variates
        )

        updated_client_control_variates = self.compute_updated_control_variates(
            local_steps, delta_model_weights, delta_control_variates
        )
        self.client_control_variates_updates = self.compute_parameters_delta(
            updated_client_control_variates, self.client_control_variates
        )

        # c_i = c_i^plus
        self.client_control_variates = updated_client_control_variates

    def modify_grad(self) -> None:
        """
        Modifies the gradient of the local model to correct for client drift.
        To be called after the gradients have been computed on a batch of data.
        Updates not applied to params until step is called on optimizer.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None

        model_params_with_grad = [
            model_params for model_params in self.model.parameters() if model_params.requires_grad
        ]

        for param, client_cv, server_cv in zip(
            model_params_with_grad, self.client_control_variates, self.server_control_variates
        ):
            assert param.grad is not None
            tensor_type = param.grad.dtype
            server_cv_tensor = torch.from_numpy(server_cv).type(tensor_type)
            client_cv_tensor = torch.from_numpy(client_cv).type(tensor_type)
            update = server_cv_tensor.to(self.device) - client_cv_tensor.to(self.device)
            param.grad += update

    def compute_parameters_delta(self, params_1: NDArrays, params_2: NDArrays) -> NDArrays:
        """
        Computes elementwise difference of two lists of NDarray
        where elements in params_2 are subtracted from elements in params_1
        """
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]

        return parameter_delta

    def compute_updated_control_variates(
        self, local_steps: int, delta_model_weights: NDArrays, delta_control_variates: NDArrays
    ) -> NDArrays:
        """
        Computes the updated local control variates according to option 2 in Equation 4 of paper
        """

        # coef = 1 / (K * eta_l)
        scaling_coeffient = 1 / (local_steps * self.learning_rate)

        # c_i^plus = c_i - c + 1/(K*lr) * (x - y_i)
        updated_client_control_variates = [
            delta_control_variate + scaling_coeffient * delta_model_weight
            for delta_control_variate, delta_model_weight in zip(delta_control_variates, delta_model_weights)
        ]
        return updated_client_control_variates

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[Losses, Dict[str, torch.Tensor]]:
        # Clear gradients from optimizer if they exist
        self.optimizer.zero_grad()

        # Get predictions and compute loss
        preds, features = self.predict(input)
        losses = self.compute_loss(preds, features, target)

        # Calculate backward pass, modify grad to account for client drift, update params
        losses.backward.backward()
        self.modify_grad()
        self.optimizer.step()

        return losses, preds

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert self.model is not None
        model_size = len(self.model.state_dict())
        parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        return parameter_exchanger

    def update_after_train(self, local_steps: int, loss_dict: Dict[str, float]) -> None:
        """
        Called after training with the number of local_steps performed over the FL round and
        the corresponding loss dictionary.
        """
        self.update_control_variates(local_steps)


class DPScaffoldClient(ScaffoldClient, InstanceLevelPrivacyClient):  # type: ignore
    """
    Federated Learning client for Instance Level Differentially Private Scaffold strategy

    Implemented as specified in https://arxiv.org/abs/2111.09278
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        ScaffoldClient.__init__(
            self,
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )

        InstanceLevelPrivacyClient.__init__(
            self,
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
