from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.adaptive_drift_constraint_client import AdaptiveDriftConstraintClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class DittoClient(AdaptiveDriftConstraintClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
    ) -> None:
        """
        This client implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization. The idea is that we want to train personalized versions of the global model for each client.
        So we simultaneously train a global model that is aggregated on the server-side and use those weights to also
        constrain the training of a local model. The constraint for this local model is identical to the FedProx loss.

        NOTE: lambda, the drift loss weight, is initially set and potentially adapted by the server akin to the
        heuristic suggested in the original FedProx paper. Adaptation is optional and can be disabled in the
        corresponding strategy used by the server

        Args:
            data_path (Path): path to the data to be used to load the data for
                client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and
                predictions of the client model
            device (torch.device): Device indicator for where to send the model,
                batches, labels etc. Often 'cpu' or 'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and
                compute the losses over each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer
                module defining when and how to do checkpointing during client-side
                training. No checkpointing is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
            progress_bar (bool): Whether or not to display a progress bar during client training and validation.
                Uses tqdm. Defaults to False
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
            reporters=reporters,
            progress_bar=progress_bar,
        )
        self.global_model: nn.Module

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys 'global' and 'local' respectively.

        Args:
            config (Config): The config from the server.
        """
        raise NotImplementedError(
            "User Clients must define a function that returns a Dict[str, Optimizer] with keys 'global' and 'local' "
            "defining separate optimizers for the global and local models of Ditto."
        )

    def set_optimizer(self, config: Config) -> None:
        """
        Ditto requires an optimizer for the global model and one for the local model. This function simply ensures that
        the optimizers setup by the user have the proper keys and that there are two optimizers.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and set(("global", "local")) == set(optimizers.keys())
        self.optimizers = optimizers

    def get_global_model(self, config: Config) -> nn.Module:
        # The global model should be the same architecture as the local model so
        # we reuse the get_model call. We explicitly send the model to the desired device. This is idempotent.
        return self.get_model(config).to(self.device)

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. In this class, this function simply adds the additional step of
        setting up the global model.

        Args:
            config (Config): The config from the server.
        """
        # Need to setup the global model here as well. It should be the same architecture as the local model.
        self.global_model = self.get_global_model(config)
        # The rest of the setup is the same
        super().setup_client(config)

    def get_parameters(self, config: Config) -> NDArrays:
        """
        For Ditto, we transfer the GLOBAL model weights to the server to be aggregated. The local model weights stay
        with the client.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: GLOBAL model weights to be sent to the server for aggregation
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
            assert self.global_model is not None and self.parameter_exchanger is not None
            # NOTE: the global model weights are sent to the server here.
            global_model_weights = self.parameter_exchanger.push_parameters(self.global_model, config=config)

            # Weights and training loss sent to server for aggregation
            # Training loss sent because server will decide to increase or decrease the penalty weight, if adaptivity
            # is turned on
            packed_params = self.parameter_exchanger.pack_parameters(global_model_weights, self.loss_for_adaptation)
            return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        Assumes that the parameters being passed contain model parameters concatenated with a penalty weight. They are
        unpacked for the clients to use in training. The parameters being passed are to be routed to the global model.
        In the first fitting round, we assume the both the global and local models are being initialized and use
        the FullParameterExchanger() to initialize both sets of model weights to the same parameters.
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

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        """
        If this is the first time we're initializing the model weights, we initialize both the global and the local
        weights together.

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
        self.parameter_exchanger.pull_parameters(parameters, self.global_model, config)

    def set_initial_global_tensors(self) -> None:
        # Saving the initial GLOBAL MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the Ditto local update penalty term.
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.global_model.parameters()
        ]

    def update_before_train(self, current_server_round: int) -> None:
        """
        Procedures that should occur before proceeding with the training loops for the models. In this case, we
        save the global models parameters to be used in constraining training of the local model.

        Args:
            current_server_round (int): Indicates which server round we are currently executing.
        """
        self.set_initial_global_tensors()

        # Need to also set the global model to train mode before any training begins.
        self.global_model.train()

        super().update_before_train(current_server_round)

    def train_step(self, input: TorchInputType, target: TorchTargetType) -> Tuple[TrainingLosses, TorchPredType]:
        """
        Mechanics of training loop follow from original Ditto implementation: https://github.com/litian96/ditto
        As in the implementation there, steps of the global and local models are done in tandem and for the same
        number of steps.

        Args:
            input (TorchInputType): input tensor to be run through both the global and local models. Here,
                TorchInputType is simply an alias for the union of torch.Tensor and Dict[str, torch.Tensor].
            target (TorchTargetType): target tensor to be used to compute a loss given each models outputs.

        Returns:
            Tuple[TrainingLosses, TorchPredType]: Returns relevant loss values from both the global and local
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
        self,
        input: TorchInputType,
    ) -> Tuple[TorchPredType, TorchFeatureType]:
        """
        Computes the predictions for both the GLOBAL and LOCAL models and pack them into the prediction dictionary

        Args:
            input (TorchInputType): Inputs to be fed into both models.

        Returns:
            Tuple[TorchPredType, TorchFeatureType]: A tuple in which the first element
            contains predictions indexed by name and the second element contains intermediate activations
            index by name. For Ditto, we only need the predictions, so the second dictionary is simply empty.

        Raises:
            ValueError: Occurs when something other than a tensor or dict of tensors is returned by the model
            forward.
        """
        if isinstance(input, torch.Tensor):
            global_preds = self.global_model(input)
            local_preds = self.model(input)
        elif isinstance(input, dict):
            # If input is a dictionary, then we unpack it before computing the forward pass.
            # Note that this assumes the keys of the input match (exactly) the keyword args
            # of the forward method.
            global_preds = self.global_model(**input)
            local_preds = self.model(**input)

        # Here we assume that global and local preds are simply tensors
        # TODO: Perhaps loosen this at a later date.
        assert isinstance(global_preds, torch.Tensor)
        assert isinstance(local_preds, torch.Tensor)
        return {"global": global_preds, "local": local_preds}, {}

    def compute_loss_and_additional_losses(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the local model loss and the global Ditto model loss (stored in additional losses) for reporting and
        training of the global model

        Args:
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
            features (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target (TorchTargetType): Ground truth data to evaluate predictions against.
        Returns:
            Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]; A tuple with:
                - The tensor for the model loss
                - A dictionary with `local_loss`, `global_loss` as additionally reported loss values.
        """

        # Compute global model vanilla loss
        assert "global" in preds
        global_loss = self.criterion(preds["global"], target)

        # Compute local model loss + ditto constraint term
        assert "local" in preds
        local_loss = self.criterion(preds["local"], target)

        additional_losses = {"local_loss": local_loss.clone(), "global_loss": global_loss}

        return local_loss, additional_losses

    def compute_training_loss(
        self,
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
            preds (TorchPredType): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (TorchFeatureType): Feature(s) of the model(s) indexed by name.
            target: (TorchTargetType): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and
                additional losses indexed by name. Additional losses includes each loss component and the global model
                loss tensor.
        """
        # Check that both models are in training mode
        assert self.global_model.training and self.model.training

        # local loss is stored in loss, global model loss is stored in additional losses.
        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)

        # Setting the adaptation loss to that of the local model, as its performance should dictate whether more or
        # less weight is used to constrain it to the global model (as in FedProx)
        additional_losses["loss_for_adaptation"] = additional_losses["local_loss"].clone()

        # This is the Ditto penalty loss of the local model compared with the original Global model weights, scaled
        # by drift_penalty_weight (or lambda in the original paper)
        penalty_loss = self.compute_penalty_loss()
        additional_losses["penalty_loss"] = penalty_loss.clone()

        return TrainingLosses(backward=loss + penalty_loss, additional_losses=additional_losses)

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        """
        Validate the current model on the entire validation dataset.

        Returns:
            Tuple[float, Dict[str, Scalar]]: The validation loss and a dictionary of metrics from validation.
        """
        # Set the global model to evaluate mode
        self.global_model.eval()
        return super().validate()

    def compute_evaluation_loss(
        self,
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
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name.
        """
        # Check that both models are in eval mode
        assert not self.global_model.training and not self.model.training
        return super().compute_evaluation_loss(preds, features, target)
