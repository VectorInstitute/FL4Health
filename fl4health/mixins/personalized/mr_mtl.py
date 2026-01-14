"""MR MTL Personalized Mixin."""

import copy
from logging import INFO
from typing import Any, Protocol, runtime_checkable

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.optim import Optimizer

from fl4health.mixins.adaptive_drift_constrained import (
    AdaptiveDriftConstrainedMixin,
    AdaptiveDriftConstrainedProtocol,
)
from fl4health.mixins.personalized.utils import ensure_protocol_compliance
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.typing import (
    TorchFeatureType,
    TorchPredType,
    TorchTargetType,
)


@runtime_checkable
class MrMtlPersonalizedProtocol(AdaptiveDriftConstrainedProtocol, Protocol):
    initial_global_model: torch.nn.Module | None
    initial_global_tensors: list[torch.Tensor]

    def get_global_model(self, config: Config) -> nn.Module:
        pass  # pragma: no cover


class MrMtlPersonalizedMixin(AdaptiveDriftConstrainedMixin):
    def __init__(self: MrMtlPersonalizedProtocol, *args: Any, **kwargs: Any) -> None:
        """
        This client implements the MR-MTL algorithm from MR-MTL: On Privacy and Personalization in Cross-Silo
        Federated Learning. The idea is that we want to train personalized versions of the global model for each
        client. However, instead of using a separate solver for the global model, as in Ditto, we update the initial
        global model with aggregated local models on the server-side and use those weights to also constrain the
        training of a local model. The constraint for this local model is identical to the FedProx loss. The key
        difference is that the local model is never replaced with aggregated weights. It is always local.

        **NOTE**: lambda, the drift loss weight, is initially set and potentially adapted by the server akin to the
        heuristic suggested in the original FedProx paper. Adaptation is optional and can be disabled in the
        corresponding strategy used by the server
        """
        # Initialize mixin-specific attributes
        self.initial_global_model: torch.nn.Module | None = None
        self.initial_global_tensors: list[torch.Tensor] = []

        super().__init__(*args, **kwargs)

    def get_global_model(self: MrMtlPersonalizedProtocol, config: Config) -> nn.Module:
        """
        Returns the global model on client setup to be used as a constraint for the local model during training.

        The global model should be the same architecture as the local model so we reuse the ``get_model`` call. We
        explicitly send the model to the desired device. This is idempotent.

        Args:
            config (Config): The config from the server.

        Returns:
            (nn.Module): The PyTorch model serving as the global model for Ditto
        """
        model_copy = copy.deepcopy(self.get_model(config))
        return model_copy.to(self.device)

    @ensure_protocol_compliance
    def setup_client(self: MrMtlPersonalizedProtocol, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. In this class, this function simply adds the additional step of
        setting up the global model.

        Args:
            config (Config): The config from the server.
        """
        try:
            self.initial_global_model = self.get_global_model(config)
            log(INFO, f"initial global model set: {type(self.initial_global_model).__name__}")
        except AttributeError:
            log(
                INFO,
                "Couldn't set initial global model before super().setup_client(). Will try again within that setup.",
            )
            pass
        # The rest of the setup is the same
        super().setup_client(config)  # type:ignore [safe-super]

    @ensure_protocol_compliance
    def get_optimizer(self: MrMtlPersonalizedProtocol, config: Config) -> dict[str, Optimizer]:
        """
        Implementing get_optimizer as a hook to set initial global model if not already set.

        Args:
            config (Config): The config from the server.
        """
        if self.initial_global_model is None:
            # try set it here
            self.initial_global_model = self.get_global_model(config)
            log(
                INFO,
                f"initial_global_model set: {type(self.initial_global_model).__name__} within `get_optimizer`",
            )

        return super().get_optimizer(config=config)  # type: ignore[safe-super, return-value]

    @ensure_protocol_compliance
    def set_parameters(
        self: MrMtlPersonalizedProtocol, parameters: NDArrays, config: Config, fitting_round: bool
    ) -> None:
        """
        The parameters being passed are to be routed to the initial global model to be used in a penalty term in
        training the local model. Despite the usual FL setup, we actually never pass the aggregated model to the
        **LOCAL** model. Instead, we use the aggregated model to form the MR-MTL penalty term.

        NOTE: In MR-MTL, unlike Ditto, the local model weights are not synced across clients to the initial global
        model, even in the **FIRST ROUND**.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model. It will also contain a penalty weight from the server at each round (possibly adapted)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning round is a fitting
                round or an evaluation round. Not used here.
        """
        # Make sure that the proper components exist.
        assert self.initial_global_model is not None and self.parameter_exchanger is not None

        # Route the parameters to the GLOBAL model only in MR-MTL
        log(INFO, "Setting the global model weights")
        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Lambda weight received from the server: {self.drift_penalty_weight}")

        self.parameter_exchanger.pull_parameters(server_model_state, self.initial_global_model, config)

    @ensure_protocol_compliance
    def update_before_train(self: MrMtlPersonalizedProtocol, current_server_round: int) -> None:
        assert self.initial_global_model is not None
        # Freeze the initial weights of the INITIAL GLOBAL MODEL. These are used to form the MR-MTL
        # update penalty term.
        for param in self.initial_global_model.parameters():
            param.requires_grad = False
        self.initial_global_model.eval()

        # Saving the initial GLOBAL MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the MR-MTL local update penalty term.
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.initial_global_model.parameters()
        ]

        return super().update_before_train(current_server_round)  # type: ignore[safe-super]

    @ensure_protocol_compliance
    def compute_training_loss(
        self: MrMtlPersonalizedProtocol,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the modes and ground truth data. We add to vanilla loss
        function by including Mean Regularized (MR) penalty loss which is the \\(\\ell^2\\) inner product between the
        initial global model weights and weights of the current model.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (TrainingLosses): An instance of ``TrainingLosses`` containing backward loss and additional losses indexed
                by name. Additional losses includes each loss component of the total loss.
        """
        # Check that the initial global model isn't in training mode and that the local model is in training mode
        assert self.initial_global_model is not None and not self.initial_global_model.training and self.model.training
        # Use the rest of the training loss computation from the AdaptiveDriftConstraintClient parent
        return super().compute_training_loss(preds, features, target)  # type: ignore[safe-super]

    @ensure_protocol_compliance
    def validate(
        self: MrMtlPersonalizedProtocol, include_losses_in_metrics: bool = False
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            (tuple[float, dict[str, Scalar]]): The validation loss and a dictionary of metrics from validation.
        """
        # ensure that the initial global model is in eval mode
        assert self.initial_global_model is not None and not self.initial_global_model.training
        return super().validate(include_losses_in_metrics=include_losses_in_metrics)  # type: ignore[safe-super]
