"""Ditto Personalized Mixin"""

import warnings
from logging import INFO, WARN
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer

from fl4health.clients.basic_client import BasicClient
from fl4health.mixins.adaptive_drift_constrained import AdaptiveDriftConstrainedMixin, AdaptiveDriftConstrainedProtocol
from fl4health.mixins.core_protocols import BasicClientProtocolPreSetup
from fl4health.mixins.personalized.utils import ensure_protocol_compliance
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import EvaluationLosses, TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


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

    def _extract_pred(self, kind: str, preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass  # pragma: no cover

    def safe_global_model(self) -> nn.Module:
        pass  # pragma: no cover


class DittoPersonalizedMixin(AdaptiveDriftConstrainedMixin):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        This mixin implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization. This mixin inherits from the `AdaptiveDriftConstrainedMixin`, and like that mixin,
        this should be mixed with a `BasicClient` type in order to apply the Ditto personalization method
        to that client.

        Background Context:
        The idea is that we want to train personalized versions of the global model for each client.
        So we simultaneously train a global model that is aggregated on the server-side and use those weights to also
        constrain the training of a local model. The constraint for this local model is identical to the FedProx loss.


        Raises:
            RuntimeError: If the object does not satisfy the `BasicClientProtocolPreSetup`
            then it will raise an error. This is additional validation to ensure that the mixin was
            applied to an appropriate base class.
        """
        # Initialize mixin-specific attributes
        self.global_model: torch.nn.Module | None = None

        # Call parent's init
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # if a parent class doesn't take args/kwargs
            super().__init__()

        if not isinstance(self, BasicClientProtocolPreSetup):
            raise RuntimeError("This object needs to satisfy `BasicClientProtocolPreSetup`.")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """This method is called when a class inherits from AdaptiveMixin"""
        super().__init_subclass__(**kwargs)

        # Skip check for other mixins
        if cls.__name__.endswith("Mixin"):
            return

        # Skip validation for dynamically created classes
        if hasattr(cls, "_dynamically_created"):
            return

        # Check at class definition time if the parent class satisfies BasicClientProtocol
        for base in cls.__bases__:
            if base is not DittoPersonalizedMixin and issubclass(base, BasicClient):
                return

        # If we get here, no compatible base was found
        msg = (
            f"Class {cls.__name__} inherits from DittoPersonalizedMixin but none of its other "
            f"base classes implement BasicClient. This may cause runtime errors."
        )
        log(WARN, msg)
        warnings.warn(
            msg,
            RuntimeWarning,
        )

    def safe_global_model(self: DittoPersonalizedProtocol) -> nn.Module:
        """Convenient accessor for the global model.

        Raises:
            ValueError: If the `global_model` attribute has not yet been set, we
            will raise an error.

        Returns:
            nn.Module: the global model if it has been set.
        """
        if self.global_model:
            return self.global_model
        raise ValueError("Cannot get global model as it not yet been set.")

    @property
    def optimizer_keys(self: DittoPersonalizedProtocol) -> list[str]:
        """Returns the optimizer keys."""
        return ["local", "global"]

    def _copy_optimizer_with_new_params(self: DittoPersonalizedProtocol, original_optimizer: Optimizer) -> Optimizer:
        OptimClass = original_optimizer.__class__
        state_dict = original_optimizer.state_dict()

        # Extract hyperparameters from param_groups
        # We only take the first group's hyperparameters, excluding 'params' and 'lr'
        param_group = state_dict["param_groups"][0]

        # store initial_lr to be used with schedulers
        initial_lr = param_group.get("initial_lr", param_group["lr"])

        optimizer_kwargs = {k: v for k, v in param_group.items() if k not in ("params", "initial_lr")}
        assert self.global_model is not None
        global_optimizer = OptimClass(self.global_model.parameters(), **optimizer_kwargs)

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
            nn.Module: The PyTorch model serving as the global model for Ditto
        """
        return self.get_model(config).to(self.device)

    @ensure_protocol_compliance
    def get_optimizer(self: DittoPersonalizedProtocol, config: Config) -> dict[str, Optimizer]:
        if self.global_model is None:
            # try set it here
            self.global_model = self.get_global_model(config)  # is this the same config?
            log(INFO, f"global model set: {type(self.global_model).__name__} within `get_optimizer`")

        # Note that the global optimizer operates on self.global_model.parameters()
        optimizer = super().get_optimizer(config=config)  # type: ignore[safe-super]
        if isinstance(optimizer, dict):
            try:
                original_optimizer = next(el for el in optimizer.values() if isinstance(el, Optimizer))
            except StopIteration:
                raise ValueError("Unable to find an ~torch.optim.Optimizer object.")
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
            log(INFO, "Couldn't set global model before super().setup_client(). Will try again within that setup.")
            pass
        # The rest of the setup is the same
        super().setup_client(config)  # type:ignore [safe-super]
        # Need to setup the global model here as well. It should be the same architecture as the local model.

    def get_parameters(self: DittoPersonalizedProtocol, config: Config) -> NDArrays:
        """
        For Ditto, we transfer the **GLOBAL** model weights to the server to be aggregated. The local model weights
        stay with the client.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: **GLOBAL** model weights to be sent to the server for aggregation
        """
        if not self.initialized:
            log(
                INFO,
                "Setting up client and providing full model parameters to the server for initialization",
            )

            # If initialized==False, the server is requesting model parameters from which to initialize all other
            # clients. As such get_parameters is being called before fit or evaluate, so we must call
            # setup_client first.
            self.setup_client(config)

            # Need all parameters even if normally exchanging partial. Since the global and local models are the same
            # architecture, it doesn't matter which we choose as an initializer. The global and local models are set
            # to the same weights in initialize_all_model_weights
            return FullParameterExchanger().push_parameters(self.model, config=config)
        else:
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
        self: DittoPersonalizedProtocol, parameters: NDArrays, config: Config, fitting_round: bool
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
            log(INFO, "Initializing the global and local models weights for the first time")
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
            parameters (NDArrays): Model parameters to be injected into the client model
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
        Mechanics of training loop follow from original Ditto implementation: https://github.com/litian96/ditto

        As in the implementation there, steps of the global and local models are done in tandem and for the same
        number of steps.

        Args:
            input (TorchInputType): input tensor to be run through both the global and local models. Here,
                ``TorchInputType`` is simply an alias for the union of ``torch.Tensor`` and
                ``dict[str, torch.Tensor]``.
            target (TorchTargetType): target tensor to be used to compute a loss given each models outputs.

        Returns:
            tuple[TrainingLosses, TorchPredType]: Returns relevant loss values from both the global and local
            model optimization steps. The prediction dictionary contains predictions indexed a "global" and "local"
            corresponding to predictions from the global and local Ditto models for metric evaluations.
        """

        # Clear gradients from optimizers if they exist
        self.optimizers["global"].zero_grad()
        self.optimizers["local"].zero_grad()

        # Forward pass on both the global and local models
        preds, features = self.predict(input)
        target = self.transform_target(target)  # Apply transformation (Defaults to identity)

        # Compute all relevant losses
        losses = self.compute_training_loss(preds, features, target)

        # Take a step with the global model vanilla loss
        losses.additional_losses["global_loss"].backward()
        self.optimizers["global"].step()

        # Take a step with the local model using the local loss and Ditto constraint
        losses.backward["backward"].backward()
        self.optimizers["local"].step()

        # Return dictionary of predictions where key is used to name respective MetricMeters
        return losses, preds

    def predict(
        self: DittoPersonalizedProtocol,
        input: TorchInputType,
    ) -> tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the predictions for both the **GLOBAL** and **LOCAL** models and pack them into the prediction
        dictionary

        Args:
            input (TorchInputType): Inputs to be fed into both models.

        Returns:
            tuple[TorchPredType, TorchFeatureType]: A tuple in which the first element contains predictions indexed by
            name and the second element contains intermediate activations index by name. For Ditto, we only need the
            predictions, so the second dictionary is simply empty.

        Raises:
            ValueError: Occurs when something other than a tensor or dict of tensors is returned by the model
                forward.
        """

        if hasattr(self, "_predict"):
            log(INFO, "Using '_predict' to make predictions")
            global_preds, _ = self._predict(self.safe_global_model(), input)
            local_preds, _ = self._predict(self.model, input)
            log(INFO, "Successfully predicted for global and local models")
        else:
            if isinstance(input, torch.Tensor):
                global_preds = self.safe_global_model()(input)
                local_preds = self.model(input)
            elif isinstance(input, dict):
                # If input is a dictionary, then we unpack it before computing the forward pass.
                # Note that this assumes the keys of the input match (exactly) the keyword args
                # of the forward method.
                global_preds = self.safe_global_model()(**input)
                local_preds = self.model(**input)

        # Here we assume that global and local preds are simply tensors
        # TODO: Perhaps loosen this at a later date.
        # assert isinstance(global_preds, torch.Tensor)
        # assert isinstance(local_preds, torch.Tensor)
        if isinstance(global_preds, torch.Tensor) and isinstance(local_preds, torch.Tensor):
            return {"global": global_preds, "local": local_preds}, {}
        elif isinstance(global_preds, dict) and isinstance(local_preds, dict):
            retval = {f"global-{k}": v for k, v in global_preds.items()}
            retval.update(**{f"local-{k}": v for k, v in local_preds.items()})
            return retval, {}
        else:
            raise ValueError(f"Unsupported pred type: {type(global_preds)}.")

    def _extract_pred(self, kind: str, preds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if kind not in ["global", "local"]:
            raise ValueError("Unsupported kind of prediction. Must be 'global' or 'local'.")

        # filter
        retval = {k: v for k, v in preds.items() if kind in k}
        # remove prefix
        retval = {k.replace(f"{kind}-", ""): v for k, v in retval.items()}
        return retval

    def compute_loss_and_additional_losses(
        self: DittoPersonalizedProtocol,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Computes the local model loss and the global Ditto model loss (stored in additional losses) for reporting and
        training of the global model

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.
        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple with:

            - The tensor for the model loss
            - A dictionary with ``local_loss``, ``global_loss`` as additionally reported loss values.
        """

        global_preds = self._extract_pred(kind="global", preds=preds)
        local_preds = self._extract_pred(kind="local", preds=preds)

        # Compute global model vanilla loss

        if hasattr(self, "_special_compute_loss_and_additional_losses"):
            log(INFO, "Using '_special_compute_loss_and_additional_losses' to compute loss")
            global_loss, _ = self._special_compute_loss_and_additional_losses(global_preds, features, target)

            # Compute local model loss + ditto constraint term
            local_loss, _ = self._special_compute_loss_and_additional_losses(local_preds, features, target)

        else:
            global_loss = self.criterion(global_preds, target)

            # Compute local model loss + ditto constraint term
            local_loss = self.criterion(local_preds, target)

        additional_losses = {"local_loss": local_loss.clone(), "global_loss": global_loss}

        return local_loss, additional_losses

    def compute_training_loss(
        self: DittoPersonalizedProtocol,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the global and local models and ground truth data.
        For the local model we add to the vanilla loss function by including Ditto penalty loss which is the l2 inner
        product between the initial global model weights and weights of the local model. This is stored in backward
        The loss to optimize the global model is stored in the additional losses dictionary under "global_loss"

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name. All predictions included in
                dictionary will be used to compute metrics.
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: An instance of ``TrainingLosses`` containing backward loss and additional losses indexed by
            name. Additional losses includes each loss component and the global model
            loss tensor.
        """
        # Check that both models are in training mode
        assert self.safe_global_model().training and self.model.training

        # local loss is stored in loss, global model loss is stored in additional losses.
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)
        additional_losses = additional_losses or {}  # make mypy happy

        # Setting the adaptation loss to that of the local model, as its performance should dictate whether more or
        # less weight is used to constrain it to the global model (as in FedProx)
        additional_losses["loss_for_adaptation"] = additional_losses["local_loss"].clone()

        # This is the Ditto penalty loss of the local model compared with the original Global model weights, scaled
        # by drift_penalty_weight (or lambda in the original paper)
        penalty_loss = self.compute_penalty_loss()
        additional_losses["penalty_loss"] = penalty_loss.clone()

        return TrainingLosses(backward=loss + penalty_loss, additional_losses=additional_losses)

    @ensure_protocol_compliance
    def validate(
        self: DittoPersonalizedProtocol, include_losses_in_metrics: bool = False
    ) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            tuple[float, dict[str, Scalar]]: The validation loss and a dictionary of metrics from validation.
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
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: An instance of ``EvaluationLosses`` containing checkpoint loss and additional losses
            indexed by name.
        """
        # Check that both models are in eval mode
        assert self.global_model is not None and not self.global_model.training and not self.model.training
        return super().compute_evaluation_loss(preds, features, target)  # type: ignore[safe-super]
