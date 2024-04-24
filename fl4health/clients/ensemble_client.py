from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from flwr.common.typing import Config
from torch.optim import Optimizer

from fl4health.checkpointing.client_module import ClientCheckpointModule
from fl4health.clients.basic_client import BasicClient, TorchInputType
from fl4health.model_bases.ensemble_base import EnsembleModel
from fl4health.utils.losses import EvaluationLosses, LossMeterType, TrainingLosses
from fl4health.utils.metrics import Metric


class EnsembleClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[ClientCheckpointModule] = None,
    ) -> None:
        """
        This client enables the training of ensemble models in a federated manner.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpointer=checkpointer,
        )

        self.model: EnsembleModel

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True. Also perform some checks to ensure the keys of the
        optimizer dictionary are consistent with the model keys.

        Args:
            config (Config): The config from the server.
        """
        super().setup_client(config)

        assert len(self.optimizers) == len(self.model.ensemble_models)
        assert all(
            opt_key == model_key
            for opt_key, model_key in zip(sorted(self.optimizers.keys()), sorted(self.model.ensemble_models.keys()))
        )

    def set_optimizer(self, config: Config) -> None:
        """
        Method called in the the setup_client method to set optimizer attribute returned by used-defined get_optimizer.
        Ensures that the return value of get_optimizer is a dictionary since that is required for the ensemble client.

        Args:
            config (Config): The config from the server.
        """
        optimizers = self.get_optimizer(config)
        assert isinstance(optimizers, dict)
        self.optimizers = optimizers

    def train_step(
        self, input: TorchInputType, target: torch.Tensor
    ) -> Tuple[TrainingLosses, Dict[str, torch.Tensor]]:
        """
        Given a single batch of input and target data, generate predictions
        (both individual models and ensemble prediction), compute loss, update parameters and
        optionally update metrics if they exist. (ie backprop on a single batch of data).
        Assumes self.model is in train mode already. Differs from parent method in that, there are multiple losses
        that we have to do backward passes on and multiple optimizers to update parameters each train step.

        Args:
            input (TorchInputType): The input to be fed into the model.
            TorchInputType is simply an alias for the union of torch.Tensor and
            Dict[str, torch.Tensor].
            target (torch.Tensor): The target corresponding to the input.

        Returns:
            Tuple[TrainingLosses, Dict[str, torch.Tensor]]: The losses object from the train step along with
                a dictionary of any predictions produced by the model.
        """
        assert isinstance(input, torch.Tensor)
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        preds, features = self.predict(input)
        losses = self.compute_training_loss(preds, features, target)

        for loss in losses.backward.values():
            loss.backward()

        for optimizer in self.optimizers.values():
            optimizer.step()

        return losses, preds

    def compute_training_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> TrainingLosses:
        """
        Computes training loss given predictions (and potentially features) of the model and ground truth data.
        Since the ensemble client has more than one model, there are multiple backward losses that exist.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            TrainingLosses: an instance of TrainingLosses containing backward loss and additional losses
                indexed by name.
        """
        loss_dict = {}
        for key, pred in preds.items():
            loss_dict[key] = self.criterion(pred.float(), target)

        individual_model_losses = {key: loss for key, loss in loss_dict.items() if key != "ensemble-pred"}
        return TrainingLosses(backward=individual_model_losses)

    def compute_evaluation_loss(
        self,
        preds: Dict[str, torch.Tensor],
        features: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> EvaluationLosses:
        """
        Computes evaluation loss given predictions (and potentially features) of the model and ground truth data.
        Since the ensemble client has more than one model, there are multiple backward losses that exist.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction(s) of the model(s) indexed by name. Anything stored
                in preds will be used to compute metrics.
            features: (Dict[str, torch.Tensor]): Feature(s) of the model(s) indexed by name.
            target: (torch.Tensor): Ground truth data to evaluate predictions against.

        Returns:
            EvaluationLosses: an instance of EvaluationLosses containing checkpoint loss and additional losses
                indexed by name.
        """
        loss_dict = {}
        for key, pred in preds.items():
            loss_dict[key] = self.criterion(pred.float(), target)

        checkpoint_loss = loss_dict["ensemble-pred"]
        return EvaluationLosses(checkpoint=checkpoint_loss)

    def get_optimizer(self, config: Config) -> Dict[str, Optimizer]:
        """
        Method to be defined by user that returns dictionary of optimizers with keys corresponding to the
        keys of the models in EnsembleModel that the optimizer applies too.

        Args:
            config (Config): The config sent from the server.

        Returns:
            Dict[str, Optimizer]: An optimizer or dictionary of optimizers to
            train model.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        raise NotImplementedError
