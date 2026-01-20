"""Ditto Personalized Mixin."""

import copy
from logging import ERROR, INFO, WARN
from typing import Any, Protocol, cast, runtime_checkable

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
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import EvaluationLosses, TrainingLosses
from fl4health.utils.typing import (
    TorchFeatureType,
    TorchInputType,
    TorchPredType,
    TorchTargetType,
)


@runtime_checkable
class DittoPersonalizedProtocol(AdaptiveDriftConstrainedProtocol, Protocol):
    global_model: torch.nn.Module | None
    optimizer_keys: list[str]

    def get_global_model(self, config: Config) -> nn.Module:
        pass  # pragma: no cover

    def _copy_optimizer_with_new_params(self, original_optimizer: Optimizer) -> Optimizer:
        pass  # pragma: no cover

    def set_initial_global_tensors(self) -> None:
        pass  # pragma: no cover

    def safe_global_model(self) -> nn.Module:
        pass  # pragma: no cover


class DittoPersonalizedMixin(AdaptiveDriftConstrainedMixin):
    def __init__(self: DittoPersonalizedProtocol, *args: Any, **kwargs: Any) -> None:
        """
        This mixin implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization. This mixin inherits from the ``AdaptiveDriftConstrainedMixin``, and like that mixin,
        this should be mixed with a ``FlexibleClient`` type in order to apply the Ditto personalization method
        to that client.

        Background Context:

        The idea is that we want to train personalized versions of the global model for each client. So we
        simultaneously train a global model that is aggregated on the server-side and use those weights to also
        constrain the training of a local model. The constraint for this local model is identical to the FedProx loss.


        Raises:
            RuntimeError: If the object does not satisfy the ``FlexibleClientProtocolPreSetup`` then it will raise an
                error. This is additional validation to ensure that the mixin was applied to an appropriate base class.
        """
        # Initialize mixin-specific attributes
        self.global_model: torch.nn.Module | None = None

        super().__init__(*args, **kwargs)

    def safe_global_model(self: DittoPersonalizedProtocol) -> nn.Module:
        """
        Convenient accessor for the global model.

        Raises:
            ValueError: If the ``global_model`` attribute has not yet been set, we will raise an error.

        Returns:
            (nn.Module): the global model if it has been set.
        """
        if self.global_model:
            return self.global_model
        raise ValueError("Cannot get global model as it not yet been set.")

    @property
    def optimizer_keys(self: DittoPersonalizedProtocol) -> list[str]:
        """
        Property for optimizer keys.

        Returns:
            (list[str]): list of keys for the optimizers dictionary.
        """
        return ["local", "global"]

    def _copy_optimizer_with_new_params(self: DittoPersonalizedProtocol, original_optimizer: Optimizer) -> Optimizer:
        """
        Helper method to make a copy of the original optimizer for the global model.

        Args:
            original_optimizer (Optimizer): original optimizer of the underlying `FlexibleClient`.

        Returns:
            (Optimizer): a copy of the original optimizer to be used by the global model.
        """
        optim_class = original_optimizer.__class__
        state_dict = original_optimizer.state_dict()

        # Extract hyperparameters from param_groups
        # We only take the first group's hyperparameters, excluding 'params' and 'lr'
        param_group = state_dict["param_groups"][0]

        # store initial_lr to be used with schedulers
        try:
            initial_lr = param_group["initial_lr"]
        except KeyError:
            if "lr" in original_optimizer.defaults:
                initial_lr = original_optimizer.defaults["lr"]
            else:
                initial_lr = 1e-3
                log(
                    WARN,
                    "Unable to get the original `lr` for the global optimizer, falling back to `1e-3`.",
                )

        optimizer_kwargs = {k: v for k, v in param_group.items() if k not in ("params", "initial_lr")}
        assert self.global_model is not None
        global_optimizer = optim_class(self.global_model.parameters(), **optimizer_kwargs)

        # maintain initial_lr for schedulers
        for param_group in global_optimizer.param_groups:
            param_group["initial_lr"] = initial_lr

        return global_optimizer

    def get_global_model(self: DittoPersonalizedProtocol, config: Config) -> nn.Module:
        """
        Returns the global model to be used during Ditto training and as a constraint for the local model.

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
    def get_optimizer(self: DittoPersonalizedProtocol, config: Config) -> dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys "global" and "local" respectively.

        Args:
            config (Config): The config from the server.
        """
        if self.global_model is None:
            # try set it here
            self.global_model = self.get_global_model(config)  # is this the same config?
            log(
                INFO,
                f"global model set: {type(self.global_model).__name__} within `get_optimizer`",
            )

        # Note that the global optimizer operates on self.global_model.parameters()
        optimizer = super().get_optimizer(config=config)  # type: ignore[safe-super]
        if isinstance(optimizer, dict):
            try:
                original_optimizer = next(el for el in optimizer.values() if isinstance(el, Optimizer))
            except StopIteration as e:
                log(ERROR, "Unable to find an ~torch.optim.Optimizer object.")
                raise e
        elif isinstance(optimizer, Optimizer):
            original_optimizer = optimizer
        else:
            raise ValueError("`super().get_optimizer()` returned an invalid type.")

        global_optimizer = self._copy_optimizer_with_new_params(original_optimizer)
        return {"local": original_optimizer, "global": global_optimizer}

    def set_optimizer(self: DittoPersonalizedProtocol, config: Config) -> None:
        """
        Ditto requires an optimizer for the global model and one for the local model. This function simply ensures that
        the optimizers setup by the user have the proper keys and that there are two optimizers.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and set(self.optimizer_keys) == set(optimizers.keys())
        self.optimizers = optimizers

    @ensure_protocol_compliance
    def setup_client(self: DittoPersonalizedProtocol, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. In this class, this function simply adds the additional step of
        setting up the global model.

        Args:
            config (Config): The config from the server.
        """
        try:
            self.global_model = self.get_global_model(config)
            log(INFO, f"global model set: {type(self.global_model).__name__}")
        except AttributeError:
            log(
                INFO,
                "Couldn't set global model before super().setup_client(). Will try again within that setup.",
            )
            pass
        # The rest of the setup is the same
        super().setup_client(config)  # type:ignore [safe-super]

    def get_parameters(self: DittoPersonalizedProtocol, config: Config) -> NDArrays:
        """
        For Ditto, we transfer the **GLOBAL** model weights to the server to be aggregated. The local model weights
        stay with the client.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            (NDArrays): **GLOBAL** model weights to be sent to the server for aggregation.
        """
        if not self.initialized:
            return self.setup_client_and_return_all_model_parameters(config)

        # NOTE: the global model weights are sent to the server here.
        if self.global_model is None:
            raise ValueError("Unable to get parameters with unset global model.")
        global_model_weights = self.parameter_exchanger.push_parameters(self.global_model, config=config)

        # Weights and training loss sent to server for aggregation
        # Training loss sent because server will decide to increase or decrease the penalty weight, if adaptivity
        # is turned on
        packed_params = self.parameter_exchanger.pack_parameters(global_model_weights, self.loss_for_adaptation)
        log(INFO, "Successfully packed parameters of global model")
        return packed_params

    @ensure_protocol_compliance
    def set_parameters(
        self: DittoPersonalizedProtocol,
        parameters: NDArrays,
        config: Config,
        fitting_round: bool,
    ) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with a penalty weight. They are
        unpacked for the clients to use in training. The parameters being passed are to be routed to the global model.
        In the first fitting round, we assume the both the global and local models are being initialized and use
        the ``FullParameterExchanger()`` to initialize both sets of model weights to the same parameters.

        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model (global model for all but the first step of Ditto). These should also include a penalty weight
                from the server that needs to be unpacked.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning
                round is a fitting round or an evaluation round. This is used to help determine which parameter
                exchange should be used for pulling parameters. If the current federated learning round is the very
                first fitting round, then we initialize both the global and local Ditto models with weights sent from
                the server.
        """
        # Make sure that the proper components exist.
        assert self.global_model is not None and self.model is not None and self.parameter_exchanger is not None
        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Lambda weight received from the server: {self.drift_penalty_weight}")

        current_server_round = narrow_dict_type(config, "current_server_round", int)
        if current_server_round == 1 and fitting_round:
            log(
                INFO,
                "Initializing the global and local models weights for the first time",
            )
            self.initialize_all_model_weights(server_model_state, config)
        else:
            # Route the parameters to the GLOBAL model in Ditto after the initial stage
            log(INFO, "Setting the global model weights")
            self.parameter_exchanger.pull_parameters(server_model_state, self.global_model, config)

    def initialize_all_model_weights(self: DittoPersonalizedProtocol, parameters: NDArrays, config: Config) -> None:
        """
        If this is the first time we're initializing the model weights, we initialize both the global and the local
        weights together.

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model.
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        parameter_exchanger = cast(FullParameterExchanger, self.parameter_exchanger)
        parameter_exchanger.pull_parameters(parameters, self.model, config)
        parameter_exchanger.pull_parameters(parameters, self.safe_global_model(), config)

    def set_initial_global_tensors(self: DittoPersonalizedProtocol) -> None:
        """
        Saving the initial **GLOBAL MODEL** weights and detaching them so that we don't compute gradients with
        respect to the tensors. These are used to form the Ditto local update penalty term.
        """
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.safe_global_model().parameters()
        ]

    @ensure_protocol_compliance
    def update_before_train(self: DittoPersonalizedProtocol, current_server_round: int) -> None:
        """
        Procedures that should occur before proceeding with the training loops for the models. In this case, we
        save the global models parameters to be used in constraining training of the local model.

        Args:
            current_server_round (int): Indicates which server round we are currently executing.
        """
        self.set_initial_global_tensors()

        # Need to also set the global model to train mode before any training begins.
        self.safe_global_model().train()

        super().update_before_train(current_server_round)  # type: ignore[safe-super]

    def train_step(
        self: DittoPersonalizedProtocol, input: TorchInputType, target: TorchTargetType
    ) -> tuple[TrainingLosses, TorchPredType]:
        """
        Mechanics of training loop follow from original Ditto implementation: https://github.com/litian96/ditto.

        As in the implementation there, steps of the global and local models are done in tandem and for the same
        number of steps.

        Args:
            input (TorchInputType): input tensor to be run through both the global and local models. Here,
                ``TorchInputType`` is simply an alias for the union of ``torch.Tensor`` and
                ``dict[str, torch.Tensor]``.
            target (TorchTargetType): target tensor to be used to compute a loss given each models outputs.

        Returns:
            (tuple[TrainingLosses, TorchPredType]): Returns relevant loss values from both the global and local
                model optimization steps. The prediction dictionary contains predictions indexed a "global" and "local"
                corresponding to predictions from the global and local Ditto models for metric evaluations.
        """
        # global
        global_losses, global_preds = self._compute_preds_and_losses(
            self.safe_global_model(), self.optimizers["global"], input, target
        )
        # local
        local_losses, local_preds = self._compute_preds_and_losses(self.model, self.optimizers["local"], input, target)
        local_loss_clone = local_losses.backward["backward"].clone()  # need a clone for later

        # take step global
        global_losses = self._apply_backwards_on_losses_and_take_step(
            self.safe_global_model(), self.optimizers["global"], global_losses
        )
        # take step local
        penalty_loss = self.compute_penalty_loss()
        local_losses.backward["backward"] = local_losses.backward["backward"] + penalty_loss
        local_losses = self._apply_backwards_on_losses_and_take_step(
            self.model, self.optimizers["local"], local_losses
        )

        # prepare return values
        additional_losses = {
            "penalty_loss": penalty_loss.clone(),
            "local_loss": local_loss_clone,
            "global_loss": global_losses.backward["backward"],
            "loss_for_adaptation": local_loss_clone.clone(),
        }
        local_losses.additional_losses = additional_losses

        # combined preds
        if isinstance(global_preds, torch.Tensor) and isinstance(local_preds, torch.Tensor):
            combined_preds = {"global": global_preds, "local": local_preds}
        elif isinstance(global_preds, dict) and isinstance(local_preds, dict):
            combined_preds = {f"global-{k}": v for k, v in global_preds.items()}
            combined_preds.update(**{f"local-{k}": v for k, v in local_preds.items()})

        return local_losses, combined_preds

    def val_step(
        self: DittoPersonalizedProtocol, input: TorchInputType, target: TorchTargetType
    ) -> tuple[EvaluationLosses, TorchPredType]:
        # global
        global_losses, global_preds = self._val_step_with_model(self.safe_global_model(), input, target)
        # local
        local_losses, local_preds = self._val_step_with_model(self.model, input, target)

        # combine
        losses = EvaluationLosses(
            local_losses.checkpoint,
            additional_losses={
                "global_loss": global_losses.checkpoint,
                "local_loss": local_losses.checkpoint,
            },
        )
        preds: TorchPredType = {}
        preds.update(**{f"global-{k}": v for k, v in global_preds.items()})
        preds.update(**{f"local-{k}": v for k, v in local_preds.items()})
        return losses, preds

    @ensure_protocol_compliance
    def validate(
        self: DittoPersonalizedProtocol, include_losses_in_metrics: bool = False
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            (tuple[float, dict[str, Scalar]]): The validation loss and a dictionary of metrics from validation.
        """
        # Set the global model to evaluate mode
        self.safe_global_model().eval()
        return super().validate(include_losses_in_metrics=include_losses_in_metrics)  # type: ignore[safe-super]

    @ensure_protocol_compliance
    def compute_evaluation_loss(
        self: DittoPersonalizedProtocol,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.
        For Ditto, we use the vanilla loss for the local model in checkpointing. However, during validation we also
        compute the global model vanilla loss.

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            (EvaluationLosses): An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
                indexed by name.
        """
        # Check that both models are in eval mode
        assert self.global_model is not None and not self.global_model.training and not self.model.training
        return super().compute_evaluation_loss(preds, features, target)  # type: ignore[safe-super]
