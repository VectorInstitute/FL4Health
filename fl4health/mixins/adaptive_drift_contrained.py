"""AdaptiveDriftConstrainedMixin"""

import warnings
from collections.abc import Sequence
from logging import INFO
from typing import Protocol, runtime_checkable

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.clients.basic_client import BasicClientProtocol
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.utils.losses import TrainingLosses
from fl4health.utils.typing import TorchFeatureType, TorchPredType, TorchTargetType


@runtime_checkable
class AdaptiveProtocol(BasicClientProtocol, Protocol):
    loss_for_adaptation: float | None
    drift_penalty_tensors: list[torch.Tensor] | None
    drift_penalty_weight: float | None

    def compute_penalty_loss(self) -> torch.Tensor: ...

    def ensure_protocol_compliance(self) -> None: ...


class AdaptiveDriftConstrainedMixin:
    def __init_subclass__(cls, **kwargs):
        """This method is called when a class inherits from AdaptiveMixin"""
        super().__init_subclass__(**kwargs)

        # Check at class definition time if the parent class satisfies BasicClientProtocol
        for base in cls.__bases__:
            if base is not AdaptiveDriftConstrainedMixin and isinstance(base, BasicClientProtocol):
                return

        # If we get here, no compatible base was found
        warnings.warn(
            f"Class {cls.__name__} inherits from AdaptiveMixin but none of its other "
            f"base classes implement BasicClientProtocol. This may cause runtime errors.",
            RuntimeWarning,
        )

    def ensure_protocol_compliance(self) -> None:
        """Call this after the object is fully initialized"""
        if not isinstance(self, BasicClientProtocol):
            raise TypeError(f"Protocol requirements not met.")

    def penalty_loss_function(self: AdaptiveProtocol) -> WeightDriftLoss:
        """Function to compute the penalty loss."""
        return WeightDriftLoss(self.device)

    def get_parameters(self: AdaptiveProtocol, config: Config) -> NDArrays:
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
        else:

            # Make sure the proper components are there
            assert (
                self.model is not None
                and self.parameter_exchanger is not None
                and self.loss_for_adaptation is not None
            )
            model_weights = self.parameter_exchanger.push_parameters(self.model, config=config)

            # Weights and training loss sent to server for aggregation. Training loss is sent because server will
            # decide to increase or decrease the penalty weight, if adaptivity is turned on.
            packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.loss_for_adaptation)
            return packed_params

    def set_parameters(self: AdaptiveProtocol, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
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

        super().set_parameters(server_model_state, config, fitting_round)

    def compute_training_loss(
        self: AdaptiveProtocol,
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

    def get_parameter_exchanger(self: AdaptiveProtocol, config: Config) -> ParameterExchanger:
        """
        Setting up the parameter exchanger to include the appropriate packing functionality.
        By default we assume that we're exchanging all parameters. Can be overridden for other behavior

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            ParameterExchanger: Exchanger that can handle packing/unpacking auxiliary server information.
        """

        return FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())

    def update_after_train(
        self: AdaptiveProtocol, local_steps: int, loss_dict: dict[str, float], config: Config
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
        super().update_after_train(local_steps, loss_dict, config)

    def compute_penalty_loss(self: AdaptiveProtocol) -> torch.Tensor:
        """
        Computes the drift loss for the client model and drift tensors

        Returns:
            torch.Tensor: Computed penalty loss tensor
        """
        # Penalty tensors must have been set for these clients.
        assert self.drift_penalty_tensors is not None

        return self.penalty_loss_function(self.model, self.drift_penalty_tensors, self.drift_penalty_weight)
