from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.losses.weight_drift_loss import WeightDriftLoss
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class FendaDittoClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        lam: float = 1.0,
        freeze_global_feature_extractor: bool = False,
    ) -> None:
        """
        This client implements the Ditto algorithm from Ditto: Fair and Robust Federated Learning Through
        Personalization with a local FENDA model. The idea is that we want to train a local FENDA model along with 
        the global model for each client. So we simultaneously train a global model that is aggregated on the server-side
        and use those weights to also constrain the training of a local FENDA model. The constraint for this 
        local FENDA model is identical to the FedProx loss.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[ClientCheckpointModule], optional): Checkpointer module defining when and how to
                do checkpointing during client-side training. No checkpointing is done if not provided. Defaults to
                None.
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
            lam (float, optional): weight applied to the Ditto drift loss. Defaults to 1.0.
            freeze_global_feature_extractor (bool, optional): Determines whether we freeze the FENDA global feature extractor during training.
                FendaDitto default is both the global and the local feature extractor in the local FENDA model will be trained, the Ditto loss
                will be calculated using the local FENDA feature extractor and the global model, otherwise we calculate using the global FENDA
                feature extractor and the global model. Defaults to False.
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
        self.global_model: SequentiallySplitExchangeBaseModel
        self.model: FendaModel
        self.ditto_drift_loss_function = WeightDriftLoss(self.device)
        self.freeze_global_feature_extractor = freeze_global_feature_extractor

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        """
        Returns a dictionary with global and local optimizers with string keys 'global' and 'local' respectively.
        """
        raise NotImplementedError(
            "User Clients must define a function that returns a Dict[str, Optimizer] with keys 'global' and 'local' "
            "defining separate optimizers for the global and local models of Ditto."
        )

    def set_optimizer(self, config: Config) -> None:
        """
        FendaDitto requires an optimizer for the global model and one for the local model. This function simply ensures that
        the optimizers setup by the user have the proper keys and that there are two optimizers.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict) and set(("global", "local")) == set(optimizers.keys())
        self.optimizers = optimizers

    def get_model(self, config: Config) -> FendaModel:
        """
        User defined method that returns FENDA model.

        Args:
            config (Config): The config from the server.

        Returns:
            FendaModel: The client FENDA model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def get_global_model(self, config: Config) -> SequentiallySplitExchangeBaseModel:
        """
        User defined method that returns a Global Sequential Model that is compatible wiFENDA model.

        Args:
            config (Config): The config from the server.

        Returns:
            FendaModel: The client FENDA model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        self.global_model = self.get_global_model(config).to(self.device)
        super().setup_client(config)

        # Check if shapes of self.global_model.feature_extractor and self.model.second_feature_extractor match
        for param1, param2 in zip(
            self.global_model.base_module.parameters(), self.model.second_feature_extractor.parameters()
        ):
            assert (
                param1.shape == param2.shape
            ), "Shapes of self.global_model.feature_extractor and self.model.second_feature_extractor do not match."

        # Check if shapes of self.model.second_feature_extractor and self.model.first_feature_extractor match
        for param1, param2 in zip(
            self.model.second_feature_extractor.parameters(), self.model.first_feature_extractor.parameters()
        ):
            assert (
                param1.shape == param2.shape
            ), "Shapes of self.model.second_feature_extractor and self.model.first_feature_extractor do not match."

    def get_parameters(self, config: Config) -> NDArrays:
        """
        For FendaDitto, we transfer the GLOBAL model weights to the server to be aggregated. The local FENDA model weights stay
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
        The parameters being passed are to be routed to the global model copied to the global feature extractor
        of the local FENDA model and saved as the initial global model tensors to be used in a penalty term in 
        training the local model. In the first fitting round, we assume the both the global and local models
        are being initialized and use the FullParameterExchanger() to set the model weights for the global model,
        the global model feature extractor weights will be then copied to the global feature extractor of local FENDA model.
        Args:
            parameters (NDArrays): Parameters have information about model state to be added to the relevant client
                model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
            fitting_round (bool): Boolean that indicates whether the current federated learning
                round is a fitting round or an evaluation round.
                This is used to help determine which parameter exchange should be used for pulling parameters.
                A full parameter exchanger is only used if the current federated learning round is the very
                first fitting round.
        """
        assert self.global_model is not None and self.model is not None
        assert self.parameter_exchanger is not None and isinstance(self.parameter_exchanger, FullParameterExchanger)

        current_server_round = self.narrow_config_type(config, "current_server_round", int)
        if current_server_round == 1 and fitting_round:
            log(INFO, "Initializing the global and local models weights for the first time")
            self.initialize_all_model_weights(parameters, config)
        else:
            log(INFO, "Setting the global model weights")
            self.parameter_exchanger.pull_parameters(parameters, self.global_model, config)
        # GLOBAL MODEL feature extractor is given to local FENDA model
        self.model.second_feature_extractor.load_state_dict(
            self.global_model.base_module.state_dict()
        ) 

    def update_before_train(self, current_server_round: int) -> None:
        # Saving the initial weights GLOBAL MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the Ditto local update penalty term.
        self.initial_global_tensors = [
            initial_layer_weights.detach().clone()
            for layer_name, initial_layer_weights in self.global_model.state_dict().items()
            if layer_name.startswith("base_module.")
        ]
        return super().update_before_train(current_server_round)

    def initialize_all_model_weights(self, parameters: NDArrays, config: Config) -> None:
        """
        If this is the first time we're initializing the model weights, we initialize the global weights.

        Args:
            parameters (NDArrays): Model parameters to be injected into the client model
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.
        """
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
        if self.freeze_global_feature_extractor:
            for param in self.model.second_feature_extractor.parameters():
                param.requires_grad = False
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
        if self.freeze_global_feature_extractor:
            for param in self.model.second_feature_extractor.parameters():
                param.requires_grad = False
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
            global_preds, _ = self.global_model(input)
            local_preds, _ = self.model(input)
        elif isinstance(input, dict):
            # If input is a dictionary, then we unpack it before computing the forward pass.
            # Note that this assumes the keys of the input match (exactly) the keyword args
            # of the forward method.
            global_preds, _ = self.global_model(**input)
            local_preds, _ = self.model(**input)
        else:
            raise TypeError(""""input" must be of type torch.Tensor or Dict[str, torch.Tensor].""")

        global_preds = global_preds["prediction"]
        local_preds = local_preds["prediction"]
        # Here we assume that global and local preds are simply tensors
        # TODO: Perhaps loosen this at a later date.
        assert isinstance(global_preds, torch.Tensor)
        assert isinstance(local_preds, torch.Tensor)
        return {"global": global_preds, "local": local_preds}, {}

    def compute_loss_and_additional_losses(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes the local model loss and any additional losses given predictions of the model and ground truth data.
        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.
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

        additional_losses = {"local_loss": local_loss, "global_loss": global_loss}

        return local_loss.clone(), additional_losses

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the global and local models and ground truth data.
        For the local model we add to the vanilla loss function by including Ditto penalty loss which is the l2 inner
        product between the initial global model feature extractor weights and feature extractor weights of the local model.
        This is stored in "backward". The loss to optimize the global model is stored in the additional losses dictionary
        under "global_loss".

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

        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)

        # Compute ditto drift loss
        if self.freeze_global_feature_extractor:
            ditto_local_loss = self.ditto_drift_loss_function(
                self.model.first_feature_extractor, self.initial_global_tensors, self.lam
            )
        else:
            ditto_local_loss = self.ditto_drift_loss_function(
                self.model.second_feature_extractor, self.initial_global_tensors, self.lam
            )
        additional_losses = additional_losses or {}
        additional_losses["ditto_loss"] = ditto_local_loss.clone()

        return TrainingLosses(backward=loss + ditto_local_loss, additional_losses=additional_losses)

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
        For FendaDitto, we use the vanilla loss for the local model in checkpointing. However, during validation we also
        compute the global model vanilla loss.

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
        return super().compute_evaluation_loss(preds, features, target)
