"""AdaptiveDriftConstrainedMixin."""

import warnings
from logging import INFO, WARN
from typing import Any, Protocol, runtime_checkable

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.clients.basic_client import BasicClient
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.mixins.core_protocols import BasicClientProtocol, BasicClientProtocolPreSetup
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchPredType, TorchTargetType


@runtime_checkable
class AdaptiveDriftConstrainedProtocol(BasicClientProtocol, Protocol):
    loss_for_adaptation: float
    drift_penalty_tensors: list[torch.Tensor] | None
    drift_penalty_weight: float | None
    penalty_loss_function: WeightDriftLoss
    parameter_exchanger: FullParameterExchangerWithPacking[float]

    def compute_penalty_loss(self) -> torch.Tensor: ...  # noqa: E704


class AdaptiveDriftConstrainedMixin:
    def __init__(self, *args: Any, **kwargs: Any):
        """
        Adaptive Drift Constrained Mixin.

        To be used with `~fl4health.BaseClient` in order to add the ability to compute
        losses via a constrained adaptive drift.

        Raises:
            RuntimeError: when the inheriting class does not satisfy `BasicClientProtocolPreSetup`.
        """
        # Initialize mixin-specific attributes with default values
        self.loss_for_adaptation = 0.1
        self.drift_penalty_tensors = None
        self.drift_penalty_weight = None

        # Call parent's init
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # if a parent class doesn't take args/kwargs
            super().__init__()

        # set penalty_loss_function
        if not isinstance(self, BasicClientProtocolPreSetup):
            raise RuntimeError("This object needs to satisfy `BasicClientProtocolPreSetup`.")
        self.penalty_loss_function = WeightDriftLoss(self.device)

    def __init_subclass__(cls, **kwargs: Any):
        """This method is called when a class inherits from AdaptiveDriftConstrainedMixin."""
        super().__init_subclass__(**kwargs)

        # Skip check for other mixins
        if cls.__name__.endswith("Mixin"):
            return

        # Skip validation for dynamically created classes
        if hasattr(cls, "_dynamically_created"):
            return

        # Check at class definition time if the parent class satisfies BasicClientProtocol
        for base in cls.__bases__:
            if base is not AdaptiveDriftConstrainedMixin and issubclass(base, BasicClient):
                return

        # If we get here, no compatible base was found
        msg = (
            f"Class {cls.__name__} inherits from AdaptiveDriftConstrainedMixin but none of its other "
            f"base classes is a BasicClient. This may cause runtime errors."
        )
        log(WARN, msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    def get_parameters(self: AdaptiveDriftConstrainedProtocol, config: Config) -> NDArrays:
        """
        Packs the parameters and training loss into a single ``NDArrays`` to be sent to the server for aggregation. If
        the client has not been initialized, this means the server is requesting parameters for initialization and
        just the model parameters are sent. When using the ``FedAvgWithAdaptiveConstraint`` strategy, this should not
        happen, as that strategy requires server-side initialization parameters. However, other strategies may handle
        this case.

        Args:
            config (Config): Configurations to allow for customization of this functions behavior

        Returns:
            NDArrays: Parameters and training loss packed together into a list of numpy arrays to be sent to the server
        """
        if not self.initialized:
            log(INFO, "Setting up client and providing full model parameters to the server for initialization")

            # If initialized is False, the server is requesting model parameters from which to initialize all other
            # clients. As such get_parameters is being called before fit or evaluate, so we must call
            # setup_client first.
            self.setup_client(config)

            # Need all parameters even if normally exchanging partial
            return FullParameterExchanger().push_parameters(self.model, config=config)

        # Make sure the proper components are there
        assert self.model is not None and self.parameter_exchanger is not None and self.loss_for_adaptation is not None
        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and training loss sent to server for aggregation. Training loss is sent because server will
        # decide to increase or decrease the penalty weight, if adaptivity is turned on.
        return self.parameter_exchanger.pack_parameters(model_weights, self.loss_for_adaptation)

    def set_parameters(
        self: AdaptiveDriftConstrainedProtocol, parameters: NDArrays, config: Config, fitting_round: bool
    ) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with a penalty weight. They are
        unpacked for the clients to use in training. In the first fitting round, we assume the full model is being
        initialized and use the ``FullParameterExchanger()`` to set all model weights.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model and also the penalty weight to be applied during training.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is a fitting
                round or an evaluation round. This is used to help determine which parameter exchange should be used
                for pulling parameters. A full parameter exchanger is always used if the current federated learning
                round is the very first fitting round.
        """
        assert self.model is not None and self.parameter_exchanger is not None

        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Penalty weight received from the server: {self.drift_penalty_weight}")

        super().set_parameters(server_model_state, config, fitting_round)  # type: ignore[safe-super]

    def compute_training_loss(
        self: AdaptiveDriftConstrainedProtocol,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions of the model and ground truth data. Adds to objective by including
        penalty loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. All predictions included in
                dictionary will be used to compute metrics.
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: An instance of ``TrainingLosses`` containing backward loss and additional losses indexed
            by name. Additional losses includes penalty loss.
        """
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        if additional_losses is None:
            additional_losses = {}

        additional_losses["loss"] = loss.clone()
        # adding the vanilla loss to the additional losses to be used by update_after_train for potential adaptation
        additional_losses["loss_for_adaptation"] = loss.clone()

        # Compute the drift penalty loss and store it in the additional losses dictionary.
        penalty_loss = self.compute_penalty_loss()
        additional_losses["penalty_loss"] = penalty_loss.clone()

        return TrainingLosses(backward=loss + penalty_loss, additional_losses=additional_losses)

    def get_parameter_exchanger(self: AdaptiveDriftConstrainedProtocol, config: Config) -> ParameterExchanger:
        """
        Setting up the parameter exchanger to include the appropriate packing functionality.
        By default we assume that we're exchanging all parameters. Can be overridden for other behavior.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            ParameterExchanger: Exchanger that can handle packing/unpacking auxiliary server information.
        """
        return FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    def update_after_train(
        self: AdaptiveDriftConstrainedProtocol, local_steps: int, loss_dict: dict[str, float], config: Config
    ) -> None:
        """
        Called after training with the number of ``local_steps`` performed over the FL round and the corresponding loss
        dictionary. We use this to store the training loss that we want to use to adapt the penalty weight parameter
        on the server side.

        Args:
            local_steps (int): The number of steps so far in the round in the local training.
            loss_dict (dict[str, float]): A dictionary of losses from local training.
            config (Config): The config from the server
        """
        assert "loss_for_adaptation" in loss_dict
        # Store current loss which is the vanilla loss without the penalty term added in
        self.loss_for_adaptation = loss_dict["loss_for_adaptation"]
        super().update_after_train(local_steps, loss_dict, config)  # type: ignore[safe-super]

    def compute_penalty_loss(self: AdaptiveDriftConstrainedProtocol) -> torch.Tensor:
        """
        Computes the drift loss for the client model and drift tensors.

        Returns:
            torch.Tensor: Computed penalty loss tensor
        """
        # Penalty tensors must have been set for these clients.
        assert self.drift_penalty_tensors is not None

        return self.penalty_loss_function(self.model, self.drift_penalty_tensors, self.drift_penalty_weight)


def apply_adaptive_drift_to_client(client_base_type: type[BasicClient]) -> type[BasicClient]:
    """Dynamically create an adapted client class.

    Args:
        client_base_type (type[BasicClient]): The class to be mixed.

    Returns:
        type[BasicClient]: A basic client that has been mixed with `AdaptiveDriftConstrainedMixin`.
    """
    return type(
        f"AdaptiveDrift{client_base_type.__name__}",
        (
            AdaptiveDriftConstrainedMixin,
            client_base_type,
        ),
        {
            # Special flag to bypass validation
            "_dynamically_created": True
        },
    )
