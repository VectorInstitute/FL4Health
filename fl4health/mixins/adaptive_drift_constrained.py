"""AdaptiveDriftConstrainedMixin."""

from logging import INFO, WARNING
from typing import Any, Protocol, runtime_checkable

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.clients.flexible.base import FlexibleClient
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.mixins.base import BaseFlexibleMixin
from fl4health.mixins.core_protocols import FlexibleClientProtocol
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.typing import TorchInputType, TorchPredType, TorchTargetType


@runtime_checkable
class AdaptiveDriftConstrainedProtocol(FlexibleClientProtocol, Protocol):
    loss_for_adaptation: float
    drift_penalty_tensors: list[torch.Tensor] | None
    drift_penalty_weight: float | None
    penalty_loss_function: WeightDriftLoss
    parameter_exchanger: FullParameterExchangerWithPacking[float]

    def compute_penalty_loss(self) -> torch.Tensor: ...

    def setup_client_and_return_all_model_parameters(self, config: Config) -> NDArrays: ...


class AdaptiveDriftConstrainedMixin(BaseFlexibleMixin):
    def __init__(self: AdaptiveDriftConstrainedProtocol, *args: Any, **kwargs: Any):
        """
        Adaptive Drift Constrained Mixin.

        To be used with ``~fl4health.BaseClient`` in order to add the ability to compute
        losses via a constrained adaptive drift.

        **NOTE**: Rather than using ``AdaptiveDriftConstraintClient``, if a client subclasses
        ``FlexibleClient``, than this mixin could be used on that subclass to implement the
        adaptive drift constraint.
        """
        # Initialize mixin-specific attributes with default values
        self.loss_for_adaptation = 0.1
        self.drift_penalty_tensors = None
        self.drift_penalty_weight = None

        super().__init__(*args, **kwargs)

        self.penalty_loss_function = WeightDriftLoss(self.device)

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
            (NDArrays): Parameters and training loss packed together into a list of numpy arrays to be sent to the
                server.
        """
        if not self.initialized:
            return self.setup_client_and_return_all_model_parameters(config)

        # Make sure the proper components are there
        assert self.model is not None and self.parameter_exchanger is not None and self.loss_for_adaptation is not None
        model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

        # Weights and training loss sent to server for aggregation. Training loss is sent because server will
        # decide to increase or decrease the penalty weight, if adaptivity is turned on.
        return self.parameter_exchanger.pack_parameters(model_weights, self.loss_for_adaptation)

    def setup_client_and_return_all_model_parameters(
        self: AdaptiveDriftConstrainedProtocol, config: Config
    ) -> NDArrays:
        """
        Function used to setup the client using the provided configuration and then exact all model parameters from
        ``self.model`` and return them. This function is used as a helper for ``get_parameters`` when the client
        has yet to be initialized.

        Args:
            config (Config): Configuration to be used  in setting up the client.

        Returns:
            (NDArrays): All parameters associated with the ``self.model`` property of the client.
        """
        log(INFO, "Setting up client and providing full model parameters to the server for initialization")
        if not config:
            log(
                WARNING,
                (
                    "This client has not yet been initialized and the config is empty. This may cause unexpected "
                    "failures, as setting up a client typically requires several configuration parameters, "
                    "including batch_size and current_server_round."
                ),
            )

        # If initialized is False, the server is requesting model parameters from which to initialize all other
        # clients. As such get_parameters is being called before fit or evaluate, so we must call
        # setup_client first.
        self.setup_client(config)

        # Need all parameters even if normally exchanging partial
        return FullParameterExchanger().push_parameters(self.model, config=config)

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

    def train_step(
        self: AdaptiveDriftConstrainedProtocol, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]:
        losses, preds = self._compute_preds_and_losses(self.model, self.optimizers["global"], input, target)
        loss_clone = losses.backward["backward"].clone()

        # apply penalty
        penalty_loss = self.compute_penalty_loss()
        losses.backward["backward"] = losses.backward["backward"] + penalty_loss
        losses = self._apply_backwards_on_losses_and_take_step(self.model, self.optimizers["global"], losses)

        # prepare return values
        additional_losses = {
            "penalty_loss": penalty_loss.clone(),
            "local_loss": loss_clone,
            "loss_for_adaptation": loss_clone.clone(),
        }
        losses.additional_losses = additional_losses

        return losses, preds

    def get_parameter_exchanger(self: AdaptiveDriftConstrainedProtocol, config: Config) -> ParameterExchanger:
        """
        Setting up the parameter exchanger to include the appropriate packing functionality.
        By default we assume that we're exchanging all parameters. Can be overridden for other behavior.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            (ParameterExchanger): Exchanger that can handle packing/unpacking auxiliary server information.
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
            (torch.Tensor): Computed penalty loss tensor.
        """
        # Penalty tensors must have been set for these clients.
        assert self.drift_penalty_tensors is not None

        return self.penalty_loss_function(self.model, self.drift_penalty_tensors, self.drift_penalty_weight)


def apply_adaptive_drift_to_client(client_base_type: type[FlexibleClient]) -> type[FlexibleClient]:
    """
    Dynamically create an adapted client class.

    Args:
        client_base_type (type[FlexibleClient]): The class to be mixed.

    Returns:
        (type[FlexibleClient]): A basic client that has been mixed with `AdaptiveDriftConstrainedMixin`.
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
