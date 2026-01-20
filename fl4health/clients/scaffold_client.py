import copy
from collections.abc import Sequence
from pathlib import Path

import torch
from flwr.common.typing import Config, NDArrays
from opacus.optimizers.optimizer import DPOptimizer

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.clients.instance_level_dp_client import InstanceLevelDpClient
from fl4health.metrics.base_metrics import Metric
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithControlVariates
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType, TrainingLosses


ScaffoldTrainStepOutput = tuple[torch.Tensor, torch.Tensor]


class ScaffoldClient(BasicClient):
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
        Federated Learning Client for Scaffold strategy.

        Implementation based on https://arxiv.org/pdf/1910.06378.pdf.

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
        self.learning_rate: float  # eta_l in paper
        self.client_control_variates: NDArrays | None = None  # c_i in paper
        self.client_control_variates_updates: NDArrays | None = None  # delta_c_i in paper
        self.server_control_variates: NDArrays | None = None  # c in paper
        # Scaffold require vanilla SGD as optimizer, will assert during setup_client
        self.optimizers: dict[str, torch.optim.Optimizer]

        self.server_model_weights: NDArrays | None = None  # x in paper
        self.parameter_exchanger: FullParameterExchangerWithPacking[NDArrays]

    def get_parameters(self, config: Config) -> NDArrays:
        """
        Packs the parameters and control variates into a single ``NDArrays`` to be sent to the server for aggregation.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            (NDArrays): Model parameters and control variates packed together.
        """
        if not self.initialized:
            return self.setup_client_and_return_all_model_parameters(config)

        assert self.model is not None and self.parameter_exchanger is not None

        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and control variates updates sent to server for aggregation
        # Control variates updates sent because only client has access to previous client control variate
        # Therefore it can only be computed locally
        assert self.client_control_variates_updates is not None
        return self.parameter_exchanger.pack_parameters(model_weights, self.client_control_variates_updates)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with server control variates.
        They are unpacked for the clients to use in training. If it's the first time the model is being initialized,
        we assume the full model is being initialized and use the ``FullParameterExchanger()`` to set all model
        weights.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the server control variates (initial or after aggregation)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Which fitting round (i.e. server round of fitting) that we're on.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, server_control_variates = self.parameter_exchanger.unpack_parameters(parameters)
        self.server_control_variates = server_control_variates

        super().set_parameters(server_model_state, config, fitting_round)

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
        Updates local control variates along with the corresponding updates according to the option 2 in Equation 4 in
        https://arxiv.org/pdf/1910.06378.pdf.

        To be called after weights of local model have been updated.

        Args:
            local_steps (int): Number of local steps performed during training.
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
        Modifies the gradient of the local model to correct for client drift. To be called after the gradients have
        been computed on a batch of data. Updates not applied to params until step is called on optimizer.
        """
        assert self.client_control_variates is not None
        assert self.server_control_variates is not None

        model_params_with_grad = [
            model_params for model_params in self.model.parameters() if model_params.requires_grad
        ]

        for param, client_cv, server_cv in zip(
            model_params_with_grad,
            self.client_control_variates,
            self.server_control_variates,
        ):
            assert param.grad is not None
            tensor_type = param.grad.dtype
            server_cv_tensor = torch.from_numpy(server_cv).type(tensor_type)
            client_cv_tensor = torch.from_numpy(client_cv).type(tensor_type)
            update = server_cv_tensor.to(self.device) - client_cv_tensor.to(self.device)
            param.grad += update

    def compute_parameters_delta(self, params_1: NDArrays, params_2: NDArrays) -> NDArrays:
        """
        Computes element-wise difference of two lists of ``NDarray`` where elements in ``params_2`` are subtracted from
        elements in ``params_1``.

        Each ``NDArray`` in the list of ``NDArrays`` are subtracted as

        \\[\\text{params}_{1, i} - \\text{params}_{2, i}\\]

        Args:
            params_1 (NDArrays): First set of parameters
            params_2 (NDArrays): Second set of parameters

        Returns:
            (NDArrays): \\(\\text{params}_1 - \\text{params}_2\\)
        """
        parameter_delta: NDArrays = [param_1 - param_2 for param_1, param_2 in zip(params_1, params_2)]

        return parameter_delta

    def transform_gradients(self, losses: TrainingLosses) -> None:
        """
        Hook function for model training only called after backwards pass but before optimizer step. Used to modify
        gradient to correct for client drift in Scaffold.

        Args:
            losses (TrainingLosses): losses is not used in this transformation.
        """
        self.modify_grad()

    def compute_updated_control_variates(
        self,
        local_steps: int,
        delta_model_weights: NDArrays,
        delta_control_variates: NDArrays,
    ) -> NDArrays:
        """
        Computes the updated local control variates according to option 2 in Equation 4 of paper.

        The calculation is

        \\[c_i^+ = c_i - c + \\frac{1}{(K \\cdot lr)} \\cdot (x - y_i)\\]

        where lr is the local learning rate.

        Args:
            local_steps (int): Number of local steps that were taken during local training (\\(K\\))
            delta_model_weights (NDArrays): difference between the locally trained weights and the initial weights
                prior to local training
            delta_control_variates (NDArrays): difference between local (\\(c_i\\)) and server (\\(c\\)) control
                variates \\(c_i - c\\).

        Returns:
            (NDArrays): Updated client control variates
        """
        # coef = 1 / (K * eta_l)
        scaling_coefficient = 1 / (local_steps * self.learning_rate)

        # c_i^plus = c_i - c + 1/(K*lr) * (x - y_i)
        return [
            delta_control_variate + scaling_coefficient * delta_model_weight
            for delta_control_variate, delta_model_weight in zip(delta_control_variates, delta_model_weights)
        ]

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        assert self.model is not None
        model_size = len(self.model.state_dict())
        return FullParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        r"""
        Called after training with the number of ``local_steps`` performed over the FL round and the corresponding
        loss dictionary.

        Args:
            local_steps (int): Number of local steps that were taken during local training (\(K\))
            loss_dict (dict[str, float]): dictionary of losses computed during training
            config (Config): The config from the server.
        """
        self.update_control_variates(local_steps)

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these. Then set
        initialized attribute to True. Extends the basic client to extract the learning rate from the optimizer and
        set the ``learning_rate`` attribute (used to compute updated control variates).

        Args:
            config (Config): The config from the server.
        """
        super().setup_client(config)
        if isinstance(self, DPScaffoldClient):
            assert isinstance(self.optimizers["global"], DPOptimizer)
        else:
            assert isinstance(self.optimizers["global"], torch.optim.SGD)
        self.learning_rate = self.optimizers["global"].defaults["lr"]


class DPScaffoldClient(ScaffoldClient, InstanceLevelDpClient):
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
        Federated Learning client for Instance Level Differentially Private Scaffold strategy.

        Implemented as specified in https://arxiv.org/abs/2111.09278

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
                validation. Uses ``tqdm``. Defaults to False.
            client_name (str | None, optional): n optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        ScaffoldClient.__init__(
            self,
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )

        InstanceLevelDpClient.__init__(
            self,
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
