from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class DittoClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        lam: float = 1.0,
    ) -> None:
        """
        This client implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization. The idea is that we want to train personalized versions of the global model for each client.
        So we simultaneously train a global model that is aggregated on the server-side and use those weights to also
        constrain the training of a local model. The constraint for this local model is identical to the FedProx loss.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
            lam (float, optional): weight applied to the Ditto drift loss. Defaults to 1.0.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )
        self.initial_global_tensors: List[torch.Tensor]
        self.lam = lam
        self.global_model: nn.Module
        self.ditto_loss_function = WeightDriftLoss(self.device)

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys 'global' and 'local' respectively.
        """
        raise NotImplementedError

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

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        # Need to setup the global model here as well. It should be the same architecture as the local model so
        # we reuse the get_model call. We explicitly send the model to the desired device. This is idempotent.
        self.global_model = self.get_model(config).to(self.device)
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
        assert self.global_model is not None and self.parameter_exchanger is not None
        return self.parameter_exchanger.push_parameters(self.global_model, config=config)

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        The parameters being pass are to be routed to the global model and saved as the initial global model tensors to
        be used in a penalty term in training the local model. In the first fitting round, we assume the both the
        global and local models are being initialized and use the FullParameterExchanger() to set all model weights.
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model (global model for all but the first step of Ditto)
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning
                round is a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round.
        """
        # Make sure that the proper components exist.
        assert self.global_model is not None and self.model is not None
        assert self.parameter_exchanger is not None and isinstance(self.parameter_exchanger, FullParameterExchanger)

        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        if current_server_round == 1 and fitting_round:
            log(INFO, "Initializing the global and local models weights for the first time")
            self.initialize_all_model_weights(parameters, config)
        else:
            # Route the parameters to the GLOBAL model in Ditto
            assert self.parameter_exchanger is not None
            log(INFO, "Setting the global model weights")
            self.parameter_exchanger.pull_parameters(parameters, self.global_model, config)

    def update_before_train(self, current_server_round: int) -> None:
        # Saving the initial weights GLOBAL MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the Ditto local update penalty term.
        self.initial_global_tensors = [
            initial_layer_weights.detach().clone() for initial_layer_weights in self.global_model.parameters()
        ]

        return super().update_before_train(current_server_round)

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

    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        """
        Train locally for the specified number of epochs.

        Args:
            epochs (int): The number of epochs for local training.
            current_round (Optional[int]): The current FL round.

        Returns:
            Tuple[Dict[str, float], Dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        # Need to also set the global model to train mode
        self.global_model.train()
        return super().train_by_epochs(epochs, current_round)

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.

        Returns:
            Tuple[Dict[str, float], Dict[str, Scalar]]: The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        # Need to also set the global model to train mode
        self.global_model.train()
        return super().train_by_steps(steps, current_round)

    def train_step(
        self, input: TorchInputType, target: torch.Tensor
    ) -> Tuple[TrainingLosses, Dict[str, torch.Tensor]]:
        """
        Mechanics of training loop follow from original Ditto implementation: https://github.com/litian96/ditto
        As in the implementation there, steps of the global and local models are done in tandem and for the same
        number of steps.

        Args:
            input (TorchInputType): input tensor to be run through
            both the global and local models. Here, TorchInputType is simply an alias
            for the union of torch.Tensor and Dict[str, torch.Tensor].
            target (torch.Tensor): target tensor to be used to compute a loss given each models outputs.

        Returns:
            Tuple[TrainingLosses, Dict[str, torch.Tensor]]: Returns relevant loss values from both the global and local
                model optimization steps. The prediction dictionary contains predictions indexed a "global" and "local"
                corresponding to predictions from the global and local Ditto models for metric evaluations.
        """

        # Clear gradients from optimizers if they exist
        self.optimizers["global"].zero_grad()
        self.optimizers["local"].zero_grad()

        # Forward pass on both the global and local models
        preds, features = self.predict(input)

        # Compute all relevant losses
        # NOTE: features here should be a blank dictionary, as we're not using them
        assert len(features) == 0
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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Computes the predictions for both the GLOBAL and LOCAL models and pack them into the prediction dictionary

        Args:
            input (Union[torch.Tensor, Dict[str, torch.Tensor]]): Inputs to be fed into both models.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: A tuple in which the first element
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
        else:
            raise TypeError(""""input" must be of type torch.Tensor or Dict[str, torch.Tensor].""")

        # Here we assume that global and local preds are simply tensors
        # TODO: Perhaps loosen this at a later date.
        assert isinstance(global_preds, torch.Tensor)
        assert isinstance(local_preds, torch.Tensor)
        return {"global": global_preds, "local": local_preds}, {}

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the global and local models and ground truth data.
        For the local model we add to vanilla loss function by including Ditto penalty loss which is the l2 inner
        product between the initial global model weights and weights of the local model. This is stored in backward
        The loss to optimize the global model is stored in the additional losses dictionary under "global_loss"

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in dictionary will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and
                additional losses indexed by name. Additional losses includes each loss component and the global model
                loss tensor.
        """
        # Check that both models are in training mode
        assert self.global_model.training and self.model.training

        # Compute global model vanilla loss
        assert "global" in preds
        global_loss = self.criterion(preds["global"], target)

        # Compute local model loss + ditto constraint term
        assert "local" in preds
        local_loss = self.criterion(preds["local"], target)

        # Compute ditto drift loss
        ditto_local_loss = self.ditto_loss_function(self.model, self.initial_global_tensors, self.lam)

        additional_losses = {"ditto_loss": ditto_local_loss, "local_loss": local_loss, "global_loss": global_loss}

        return TrainingLosses(backward=local_loss + ditto_local_loss, additional_losses=additional_losses)

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
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.
        For Ditto, we use the vanilla loss for the local model in checkpointing. However, during validation we also
        compute the global model vanilla loss.
        We also include a sanity check log which computes the ditto drift loss during evaluation to ensure that it
        is non-zero.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name.
        """
        # Check that both models are in eval mode
        assert not self.global_model.training and not self.model.training

        # Compute global model vanilla loss
        assert "global" in preds
        global_loss = self.criterion(preds["global"], target)

        # Compute local model vanilla loss
        assert "local" in preds
        local_loss = self.criterion(preds["local"], target)

        additional_losses = {"local_loss": local_loss, "global_loss": global_loss}

        return EvaluationLosses(checkpoint=local_loss, additional_losses=additional_losses)
