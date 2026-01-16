from __future__ import annotations

from collections.abc import Sequence
from logging import INFO
from typing import TYPE_CHECKING

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Scalar
from torch import nn


if TYPE_CHECKING:
    from fl4health.servers.base_server import FlServer

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.checkpointing.opacus_checkpointer import OpacusCheckpointer
from fl4health.checkpointing.state_checkpointer import NnUnetServerStateCheckpointer, ServerStateCheckpointer
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ExchangerType
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
    ParameterPackerWithClippingBit,
    ParameterPackerWithControlVariates,
    ParameterPackerWithLayerNames,
    SparseCooParameterPacker,
)


ModelCheckpointers = TorchModuleCheckpointer | Sequence[TorchModuleCheckpointer] | None


class BaseServerCheckpointAndStateModule:
    def __init__(
        self,
        model: nn.Module | None = None,
        parameter_exchanger: ExchangerType | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle basic model and state checkpointing on the server-side of an FL process. Unlike
        the module on the client side, this module has no concept of pre- or post-aggregation checkpointing. It only
        considers checkpointing the global server model after aggregation, perhaps based on validation statistics
        retrieved on the client side by running a federated evaluation step. Multiple model checkpointers may be
        used. For state checkpointing, which saves the state of the entire server-side FL process to help with
        FL restarts, we allow only a single checkpointer responsible for saving the state after each fit and eval
        round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            parameter_exchanger (ExchangerType | None, optional): This will facilitate routing the
                server parameters into the right components of the provided model architecture. Note that this
                exchanger and the model must match the one used for training and exchange with the servers to ensure
                parameters go to the right places. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        self.model = model
        self.parameter_exchanger = parameter_exchanger
        self.model_checkpointers = (
            [model_checkpointers] if isinstance(model_checkpointers, TorchModuleCheckpointer) else model_checkpointers
        )
        self.state_checkpointer = state_checkpointer
        if self.model_checkpointers is not None and len(self.model_checkpointers):
            # If there are model checkpointers, make sure the the model and parameter exchanger are defined.
            self._validate_model_checkpointer_components()
        self._check_if_shared_checkpoint_names()

    def _validate_model_checkpointer_components(self) -> None:
        assert self.model is not None, (
            "Checkpointer(s) is (are) defined but no model is defined to hydrate. The functionality of "
            "this class can be overridden in a child class if checkpointing without a parameter exchanger is "
            "possible and desired"
        )
        assert self.parameter_exchanger is not None, (
            "Checkpointer(s) is (are) defined but no parameter_exchanger is defined to hydrate. The functionality of "
            "this class can be overridden in a child class if checkpointing without a parameter exchanger is "
            "possible and desired"
        )

    def _check_if_shared_checkpoint_names(self) -> None:
        """
        This function is meant to throw an exception if there is an overlap in the paths to which model checkpointers
        will save model checkpoints to avoid accidental overwriting.
        """
        checkpointer_paths = (
            [checkpointer.checkpoint_path for checkpointer in self.model_checkpointers]
            if self.model_checkpointers
            else []
        )
        unique_paths = set(checkpointer_paths)

        if len(unique_paths) != len(checkpointer_paths):
            formatted_all_paths = "\n".join(checkpointer_paths)
            raise ValueError(
                "The paths of all of your checkpointers should be unique otherwise overwrites are possible and data "
                f"will be lost. The current paths are:\n{formatted_all_paths}"
            )

    def maybe_checkpoint(self, server_parameters: Parameters, loss: float, metrics: dict[str, Scalar]) -> None:
        """
        If there are model checkpointers defined in this class, we hydrate the model for checkpointing with the server
        parameters and call maybe checkpoint model on each of the checkpointers to decide whether to checkpoint based
        on the model metrics or loss and the checkpointer definitions.

        Args:
            server_parameters (Parameters): Parameters held by the server that should be injected into the model
            loss (float): The aggregated loss value obtained by the current aggregated server model.
                Potentially used by checkpointer to decide whether to checkpoint the model.
            metrics (dict[str, Scalar]): The aggregated metrics obtained by the aggregated server model. Potentially
                used by checkpointer to decide whether to checkpoint the model.
        """
        if self.model_checkpointers is not None and len(self.model_checkpointers) > 0:
            assert self.model is not None
            self._hydrate_model_for_checkpointing(server_parameters)
            for checkpointer in self.model_checkpointers:
                checkpointer.maybe_checkpoint(self.model, loss, metrics)
        else:
            log(INFO, "No model checkpointers specified. Skipping any checkpointing.")

    def _hydrate_model_for_checkpointing(self, server_parameters: Parameters) -> None:
        """
        This function is used as a means of saving the server-side model after aggregation in the FL training
        trajectory. Presently, the server only holds Flower Parameters, which are essentially just ndarrays. Without
        knowledge of a model architecture to which the arrays correspond. Thus, in the default implementation, we
        require that a torch architecture and a parameter exchanger be provided which handles mapping these numpy
        arrays into the architecture properly.

        This function may be overridden in a child class if different behavior is desired.

        **NOTE**: This function stores the weights directly in the self.model attribute

        Args:
            server_parameters (Parameters): Parameters to be injected into the torch model architecture and
            checkpointed.
        """
        assert self.model is not None, "Hydrate model for checkpoint called but self.model is None"
        assert self.parameter_exchanger is not None, (
            "Hydrate model for checkpoint called but self.parameter_exchanger is None"
        )
        model_ndarrays = parameters_to_ndarrays(server_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.model)

    def save_state(self, server: FlServer, server_parameters: Parameters) -> None:
        """
        Facilitates saving state required to restart the FL process on the server side. By default, this function
        will preserve the state of the server as defined by ``snapshot_attrs`` in ``ServerStateCheckpointer`` .
        Note that ``server_parameters`` will be hydrated and passed to the state checkpointer module to facilitate
        saving the state of the server's parameters.

        Args:
            server (FlServer): Server object from which state will be extracted and saved.
            server_parameters (Parameters): Like model checkpointers, these are the aggregated Parameters stored by
                the server representing model state. They are mapped to a torch model architecture via the
                ``_hydrate_model_for_checkpointing`` function.

        Raises:
            ValueError: Throws an error if this function is called, but no state checkpointer has been provided.
        """
        if self.state_checkpointer is not None:
            self._hydrate_model_for_checkpointing(server_parameters)
            assert self.model is not None
            self.state_checkpointer.save_server_state(server, self.model)
        else:
            raise ValueError("Attempting to save state but no state checkpointer is specified")

    def maybe_load_state(self, server: FlServer) -> Parameters | None:
        """
        Facilitates loading of any pre-existing state in the directory of the ``state_checkpointer``. If a
        ``state_checkpointer`` is defined and a checkpoint exists at its ``checkpoint_path``, this method hydrates the
        model with the saved state and returns the corresponding server Parameters. If no checkpoint exists, it logs
        this information and returns None.

        Args:
            server (FlServer): server into which checkpointed state will be loaded if a checkpoint exists

        Raises:
            ValueError: Throws an error if this function is called, but no state checkpointer has been provided.

        Returns:
            (Parameters | None): If the state checkpoint properly exists and is loaded correctly, ``server_parameters``
                is returned. Otherwise, we return a None (or throw an exception).
        """
        if self.state_checkpointer is not None:
            assert self.model is not None, (
                "Attempting to load state but self.model is None, make sure to pass the model architecture"
                " to checkpointing module"
            )
            server_model = self.state_checkpointer.maybe_load_server_state(server, self.model)
            if server_model:
                assert self.parameter_exchanger is not None
                return ndarrays_to_parameters(self.parameter_exchanger.push_parameters(server_model))
            return None
        raise ValueError("Attempting to load state, but no state checkpointer is specified")


class PackingServerCheckpointAndAndStateModule(BaseServerCheckpointAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        parameter_exchanger: FullParameterExchangerWithPacking | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to be a base class for any server-side checkpointing module that relies on unpacking
        of parameters to hydrate models for checkpointing. The specifics of the unpacking will be handled by the
        child classes of the packer within the parameter exchange.
        **NOTE**: This function ASSUMES full parameter exchange unpacking. If more complex unpacking/parameter exchange
        is used, this is not the right parent class.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            parameter_exchanger (FullParameterExchangerWithPacking | None, optional): This will facilitate routing the
                server parameters into the right components of the provided model architecture. It specifically also
                should handle any necessary unpacking of the parameters. Note that this exchanger and the model must
                match the one used for training and exchange with the servers to ensure parameters go to the right
                places. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        if parameter_exchanger is not None:
            assert isinstance(parameter_exchanger, FullParameterExchangerWithPacking), (
                "Parameter exchanger must be of based type FullParameterExchangerWithPacking"
            )
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)

    def _hydrate_model_for_checkpointing(self, server_parameters: Parameters) -> None:
        """
        This function is used as a means of saving the server-side model after aggregation in the FL training
        trajectory. Presently, the server only holds Flower Parameters, which are essentially just ndarrays. Without
        knowledge of a model architecture to which the arrays correspond. Thus, in the default implementation, we
        require that a torch architecture and a parameter exchanger be provided which handles mapping these numpy
        arrays into the architecture properly.

        This function overrides the base functionality of model hydration to insert an additional unpacking step
        using the unpacking function of the specific type of parameter exchanger.

        **NOTE**: This function stores the weights directly in the self.model attribute

        Args:
            server_parameters (Parameters): Parameters to be injected into the torch model architecture and
            checkpointed.
        """
        assert self.model is not None, "Hydrate model for checkpoint called but self.model is None"
        assert self.parameter_exchanger is not None, (
            "Hydrate model for checkpoint called but self.parameter_exchanger is None"
        )
        packed_parameters = parameters_to_ndarrays(server_parameters)
        assert isinstance(self.parameter_exchanger, FullParameterExchangerWithPacking)
        # Use the unpacking function of the parameter exchange to handle extraction of model parameters
        model_ndarrays, _ = self.parameter_exchanger.unpack_parameters(packed_parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.model)


class ScaffoldServerCheckpointAndStateModule(PackingServerCheckpointAndAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle SCAFFOLD model and state checkpointing on the server-side of an FL process.
        Unlike the module on the client side, this module has no concept of pre- or post-aggregation checkpointing.
        It only considers checkpointing the global server model after aggregation, perhaps based on validation
        statistics retrieved on the client side by running a federated evaluation step. Multiple model checkpointers
        may be used. For state checkpointing, which saves the state of the entire server-side FL process to help with
        FL restarts, we allow only a single checkpointer responsible for saving the state after each fit and eval
        round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this  checkpointer will save much more than just the model being trained. Defaults to None.
        """
        if model is not None:
            model_size = len(model.state_dict())
            parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
        else:
            parameter_exchanger = None
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)


class AdaptiveConstraintServerCheckpointAndStateModule(PackingServerCheckpointAndAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle FL flows with adaptive constraints, where the server and client communicate
        a loss weight parameter in addition to the model weights. Unlike the module on the client side, this module
        has no concept of pre- or post-aggregation checkpointing. It only considers checkpointing the global server
        model after aggregation, perhaps based on validation statistics retrieved on the client side by running a
        federated evaluation step. Multiple model checkpointers may be used. For state checkpointing, which saves the
        state of the entire server-side FL process to help with FL restarts, we allow only a single checkpointer
        responsible for saving the state after each fit and eval round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        if model is not None:
            parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
        else:
            parameter_exchanger = None
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)


class ClippingBitServerCheckpointAndStateModule(PackingServerCheckpointAndAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle FL flows with clipping bits being passed to the server along with the model
        weights. This is used for DP-FL with adaptive clipping. Unlike the module on the client side, this module
        has no concept of pre- or post-aggregation checkpointing. It only considers checkpointing the global server
        model after aggregation, perhaps based on validation statistics retrieved on the client side by running a
        federated evaluation step. Multiple model checkpointers may be used. For state checkpointing, which saves the
        state of the entire server-side FL process to help with FL restarts, we allow only a single checkpointer
        responsible for saving the state after each fit and eval round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        if model is not None:
            parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerWithClippingBit())
        else:
            parameter_exchanger = None
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)


class LayerNamesServerCheckpointAndStateModule(PackingServerCheckpointAndAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle FL flows with layer names being passed to the server along with the model
        weights. This is used for adaptive layer exchange FL. Unlike the module on the client side, this module
        has no concept of pre- or post-aggregation checkpointing. It only considers checkpointing the global server
        model after aggregation, perhaps based on validation statistics retrieved on the client side by running a
        federated evaluation step. Multiple model checkpointers may be used. For state checkpointing, which saves the
        state of the entire server-side FL process to help with FL restarts, we allow only a single checkpointer
        responsible for saving the state after each fit and eval round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        if model is not None:
            parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerWithLayerNames())
        else:
            parameter_exchanger = None
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)


class SparseCooServerCheckpointAndStateModule(PackingServerCheckpointAndAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle FL flows with parameters encoded in a sparse COO format being passed to the
        server as the model weights. This is used for adaptive parameter-wise exchange (i.e. unstructured subsets of
        parameters) . Unlike the module on the client side, this module has no concept of pre- or post-aggregation
        checkpointing. It only considers checkpointing the global server model after aggregation, perhaps based on
        validation statistics retrieved on the client side by running a federated evaluation step. Multiple model
        checkpointers may be used. For state checkpointing, which saves the state of the entire server-side FL process
        to help with FL restarts, we allow only a single checkpointer responsible for saving the state after each fit
        and eval round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained.
                Defaults to None.
        """
        if model is not None:
            parameter_exchanger = FullParameterExchangerWithPacking(SparseCooParameterPacker())
        else:
            parameter_exchanger = None
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)


class OpacusServerCheckpointAndStateModule(BaseServerCheckpointAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        parameter_exchanger: ExchangerType | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle FL flows with Opacus models where special treatment by the checkpointers is
        required. This module simply ensures the checkpointers are of the proper type before proceeding.
        Unlike the module on the client side, this module has no concept of pre- or post-aggregation checkpointing.
        It only considers checkpointing the global server model after aggregation, perhaps based on validation
        statistics retrieved on the client side by running a federated evaluation step. Multiple model checkpointers
        may be used. For state checkpointing, which saves the state of the entire server-side FL process to help with
        FL restarts, we allow only a single checkpointer responsible for saving the state after each fit and eval
        round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            parameter_exchanger (FullParameterExchangerWithPacking | None, optional): This will facilitate routing the
                server parameters into the right components of the provided model architecture. Note that this
                exchanger and the model must match the one used for training and exchange with the servers to ensure
                parameters go to the right places. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)
        self._ensure_checkpointers_are_of_opacus_type()

    def _ensure_checkpointers_are_of_opacus_type(self) -> None:
        """Helper function to ensure that the provided checkpointers are explicitly compatible with Opacus."""
        if self.model_checkpointers is not None:
            for checkpointer in self.model_checkpointers:
                assert isinstance(checkpointer, OpacusCheckpointer), (
                    "Provided checkpointers must have base class OpacusCheckpointer"
                )


class NnUnetServerCheckpointAndStateModule(BaseServerCheckpointAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        parameter_exchanger: ExchangerType | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: NnUnetServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to be used with the ``NnUnetServer`` class to handle model and state checkpointing on the
        server-side of an FL process. Unlike the module on the client side, this module has no concept of pre- or
        post-aggregation checkpointing. It only considers checkpointing the global server model after aggregation,
        perhaps based on validation statistics retrieved on the client side by running a federated evaluation step.
        Multiple model checkpointers may be used. For state checkpointing, which saves the state of the entire
        server-side FL process to help with FL restarts, we allow only a single checkpointer responsible for saving
        the state after each fit and eval round of FL.

        This implementation differs from the base class in the federated NnUnet only initializes its model after an
        initial communication phase with the clients. As such, the model is not necessarily available upon
        initialization, but may be set later.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved.

                **NOTE**: For NnUnet, this need not be set upon creation, as the model architecture may only be known
                later

                Defaults to None.

            parameter_exchanger (FullParameterExchangerWithPacking | None, optional): This will facilitate routing the
                server parameters into the right components of the provided model architecture. Note that this
                exchanger and the model must match the one used for training and exchange with the servers to ensure
                parameters go to the right places. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (NnUnetServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        super().__init__(model, parameter_exchanger, model_checkpointers, state_checkpointer)

    def _validate_model_checkpointer_components(self) -> None:
        # NOTE: We only check if the parameter exchanger is present. Model may be set later.
        assert self.parameter_exchanger is not None, (
            "Checkpointer(s) is (are) defined but no parameter_exchanger is defined to hydrate. The functionality of "
            "this class can be overridden in a child class if checkpointing without a parameter exchanger is "
            "possible and desired"
        )


class DpScaffoldServerCheckpointAndStateModule(ScaffoldServerCheckpointAndStateModule):
    def __init__(
        self,
        model: nn.Module | None = None,
        model_checkpointers: ModelCheckpointers = None,
        state_checkpointer: ServerStateCheckpointer | None = None,
    ) -> None:
        """
        This module is meant to handle DP SCAFFOLD model and state checkpointing on the server-side of an FL process.
        Unlike the module on the client side, this module has no concept of pre- or post-aggregation checkpointing.
        It only considers checkpointing the global server model after aggregation, perhaps based on validation
        statistics retrieved on the client side by running a federated evaluation step. Multiple model checkpointers
        may be used. For state checkpointing, which saves the state of the entire server-side FL process to help with
        FL restarts, we allow only a single checkpointer responsible for saving the state after each fit and eval
        round of FL.

        Args:
            model (nn.Module | None, optional): Model architecture to be saved. The module will use this architecture
                to hold the server parameters and facilitate checkpointing with the help of the parameter exchanger.
                Recall that servers only have parameters rather than torch models. So we need to know where to route
                these parameters to allow for real models to be saved. Defaults to None.
            model_checkpointers (ModelCheckpointers, optional): If defined, this checkpointer (or sequence of
                checkpointers) is used to checkpoint models based on their defined scoring function. Defaults to None.
            state_checkpointer (ServerStateCheckpointer | None, optional): If defined, this checkpointer
                will be used to preserve FL training state to facilitate restarting training if interrupted.
                Generally, this checkpointer will save much more than just the model being trained. Defaults to None.
        """
        super().__init__(model, model_checkpointers, state_checkpointer)
        self._ensure_checkpointers_are_of_opacus_type()

    def _ensure_checkpointers_are_of_opacus_type(self) -> None:
        """Helper function to ensure that the provided checkpointers are explicitly compatible with Opacus."""
        if self.model_checkpointers is not None:
            for checkpointer in self.model_checkpointers:
                assert isinstance(checkpointer, OpacusCheckpointer), (
                    "Provided checkpointers must have base class OpacusCheckpointer"
                )
