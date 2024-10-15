from logging import INFO
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.ditto_client import DittoClient
from fl4health.model_bases.fenda_base import FendaModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.losses import LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric
from fl4health.utils.parameter_extraction import check_shape_match
from fl4health.utils.typing import TorchFeatureType, TorchInputType, TorchPredType, TorchTargetType


class FendaDittoClient(DittoClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        freeze_global_feature_extractor: bool = False,
    ) -> None:
        """
        This client implements a combination of the Ditto algorithm from Ditto: Fair and Robust Federated Learning
        Through Personalization with FENDA-FL models. In this implementation, the global Ditto model consists of a
        feature extractor and classification head, where the feature extractor architecture is identical to that of
        the global and local feature extractors of the FENDA model being trained. The idea is that we want to train a
        local FENDA model along with the global model for each client. We simultaneously train a global model that is
        aggregated on the server-side and use those weights to also constrain the training of a local
        FENDA model. At the beginning of each server round, the feature extractor from globally aggregated model is
        injected into the global feature extractor of the FENDA model.

        There are two distinct modes of operation:
            If freeze_global_feature_extractor is True. The global Ditto model feature extractor SETS AND FREEZES
            weights of global FENDA feature extractor. The local components of the FENDA model are trained and an
            additional drift loss is computed between the local and global feature extractors of the FENDA model.

            If freeze_global_feature_extractor is False. The global Ditto model feature extractor INITIALIZES weights
            of the FENDA model's global feature extractor, both local and global components of FENDA are trained and
            a drift loss is calculated between Ditto global feature extractor and FENDA global feature extractor.


        The constraint for the FENDA model feature extractors discussed above uses a weight drift loss on its
        feature extraction modules.

        NOTE: Unlike FENDA, the global feature extractor of the FENDA model is NOT exchanged with the server. Rather,
        the global Ditto model is exchanged and injected at each round into the global feature extractor. If the
        global feature extractor is frozen, then only the local components of the FENDA network are trained.

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
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the client should send data to.
            progress_bar (bool): Whether or not to display a progress bar during client training and validation.
                Uses tqdm. Defaults to False

            freeze_global_feature_extractor (bool, optional): Determines whether we freeze the FENDA global feature
                extractor during training. If freeze_global_feature_extractor is False, both the global and the local
                feature extractor in the local FENDA model will be trained. Otherwise, the global feature extractor
                submodule is frozen. If freeze_global_feature_extractor is True, the Ditto loss will be calculated
                using the local FENDA feature extractor and the global model. Otherwise, the loss is calculated using
                the global FENDA feature extractor and the global model. Defaults to False.
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
        self.global_model: SequentiallySplitExchangeBaseModel
        self.model: FendaModel
        self.freeze_global_feature_extractor = freeze_global_feature_extractor

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
        raise NotImplementedError("This function must be defined in the inheriting class to use this client")

    def get_global_model(self, config: Config) -> SequentiallySplitExchangeBaseModel:
        """
        User defined method that returns a Global Sequential Model that is compatible with the local FENDA model.

        Args:
            config (Config): The config from the server.

        Returns:
            SequentiallySplitExchangeBaseModel: The global (Ditto) model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError("This function must be defined in the inheriting class to use this client")

    def _check_shape_match(self) -> None:
        """
        Checks that the defined Ditto model is compatible with the sub-components of the FENDA model and that the
        feature extractors of the FENDA model are also compatible.
        """
        # Check if shapes of global_model feature_extractor and self.model.second_feature_extractor match
        check_shape_match(
            self.global_model.base_module.parameters(),
            self.model.second_feature_extractor.parameters(),
            "Shapes of self.global_model.feature_extractor and self.model.second_feature_extractor do not match.\
                For FENDA+Ditto, these components much match exactly.",
        )

        # Check if shapes of self.model.second_feature_extractor and self.model.first_feature_extractor match
        check_shape_match(
            self.model.second_feature_extractor.parameters(),
            self.model.first_feature_extractor.parameters(),
            "Shapes of self.model.second_feature_extractor and self.model.first_feature_extractor do not match.\
                For FENDA+Ditto, these components much match exactly.",
        )

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. This function simply straps on the compatibility of the models.

        Args:
            config (Config): The config from the server.
        """
        super().setup_client(config)
        self._check_shape_match()

    def get_parameters(self, config: Config) -> NDArrays:
        """
        For FendaDitto, we transfer the GLOBAL Ditto model weights to the server to be aggregated. The local FENDA
        model weights stay with the client. The local FENDA model has a different architecture than the GLOBAL model.
        So if the client is being asked for initialization parameters, we just send the GLOBAL model to sync all GLOBAL
        models across clients AND the local FENDA model's global feature extractor.

        Args:
            config (Config): The config is sent by the FL server to allow for customization in the function if desired.

        Returns:
            NDArrays: GLOBAL model weights to be sent to the server for aggregation
        """
        if not self.initialized:
            log(INFO, "Setting up client")
            self.setup_client(config)

        assert (
            self.global_model is not None
            and self.parameter_exchanger is not None
            and self.loss_for_adaptation is not None
        )

        model_weights = self.parameter_exchanger.push_parameters(self.global_model, config=config)
        # Weights and training loss sent to server for aggregation
        # Training loss sent because server will decide to increase or decrease the penalty weight, if adaptivity
        # is turned on
        packed_params = self.parameter_exchanger.pack_parameters(model_weights, self.loss_for_adaptation)
        return packed_params

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        """
        The parameters being passed are to be routed to the global (ditto) model and copied to the global feature
        extractor of the local FENDA model and saved as the initial global model tensors to be used in a penalty term
        in training the local model. We assume the both the global and local models are being initialized and use
        a FullParameterExchanger() to set the model weights for the global model, the global model feature
        extractor weights will be then copied to the global feature extractor of local FENDA model.
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
        # Make sure that the proper components exist.
        assert self.global_model is not None and self.model is not None
        assert self.parameter_exchanger is not None and isinstance(
            self.parameter_exchanger, FullParameterExchangerWithPacking
        )

        server_model_state, self.drift_penalty_weight = self.parameter_exchanger.unpack_parameters(parameters)
        log(INFO, f"Penalty weight received from the server: {self.drift_penalty_weight}")

        self.parameter_exchanger.pull_parameters(server_model_state, self.global_model, config)
        # GLOBAL MODEL feature extractor is given to local FENDA model
        self.model.second_feature_extractor.load_state_dict(self.global_model.base_module.state_dict())

    def set_initial_global_tensors(self) -> None:
        # Saving the initial GLOBAL (DITTO) MODEL weights and detaching them so that we don't compute gradients with
        # respect to the tensors. These are used to form the Ditto local update penalty term.
        # NOTE: We are only saving the base model parameters, as these will be used to constraint a feature extractor
        # in the local FENDA model (not the full stack)
        self.drift_penalty_tensors = [
            initial_layer_weights.detach().clone()
            for initial_layer_weights in self.global_model.base_module.parameters()
        ]

    def update_before_train(self, current_server_round: int) -> None:
        # freeze the global feature extractor during training updates if desired.
        if self.freeze_global_feature_extractor:
            for param in self.model.second_feature_extractor.parameters():
                param.requires_grad = False
        return super().update_before_train(current_server_round)

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
            index by name. For Ditto+FENDA, we only need the predictions, so the second dictionary is simply empty.

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

        global_preds = global_preds["prediction"]
        local_preds = local_preds["prediction"]
        # Here we assume that global and local preds are simply tensors
        # TODO: Perhaps loosen this at a later date.
        assert isinstance(global_preds, torch.Tensor)
        assert isinstance(local_preds, torch.Tensor)
        return {"global": global_preds, "local": local_preds}, {}

    def compute_training_loss(
        self,
        preds: TorchPredType,
        features: TorchFeatureType,
        target: TorchTargetType,
    ) -> TrainingLosses:
        """
        Computes training losses given predictions of the global and local models and ground truth data.
        For the local model, we add to the vanilla loss function by including a Ditto penalty loss. This penalty
        is the L2 inner product between the initial global model feature extractor weights and the feature extractor
        weights of the local model. If the global feature extractor is not frozen, the penalty is computed using the
        global feature extractor of the local model. If it is frozen, the penalty is computed using the local feature
        extractor of the local model. This allows for flexibility in training scenarios where the feature extractors
        may differ between the global and local models. The penalty is stored in "backward". The loss to
        optimize the global model is stored in the additional losses dictionary under "global_loss".

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name.
                All predictions included in the dictionary will be used to compute metrics.
            features (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: An instance of TrainingLosses containing the backward loss and
                additional losses indexed by name. Additional losses include each loss component and the global model
                loss tensor.
        """
        # Check that both models are in training mode
        assert self.global_model.training and self.model.training

        loss, additional_losses = self.compute_loss_and_additional_losses(preds, features, target)

        if additional_losses is None:
            additional_losses = {}

        # adding the vanilla loss to the additional losses to be used by update_after_train for potential adaptation
        additional_losses["loss_for_adaptation"] = loss.clone()

        # Compute the appropriate Ditto drift loss
        if self.freeze_global_feature_extractor:
            penalty_loss = self.penalty_loss_function(
                self.model.first_feature_extractor, self.drift_penalty_tensors, self.drift_penalty_weight
            )
        else:
            penalty_loss = self.penalty_loss_function(
                self.model.second_feature_extractor, self.drift_penalty_tensors, self.drift_penalty_weight
            )
        additional_losses["penalty_loss"] = penalty_loss.clone()

        return TrainingLosses(backward=loss + penalty_loss, additional_losses=additional_losses)
