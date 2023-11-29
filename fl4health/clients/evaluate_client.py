from logging import INFO, WARNING
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.utils.losses import Losses, LossMeter, LossMeterType
from fl4health.utils.metrics import Metric, MetricManager


class EvaluateClient(NumpyFlClient):
    """
    This client implements an evaluation only flow. That is, there is no expectation of parameter exchange with the
    server past the model initialization stage. The implementing client should instantiate a global model if one is
    expected from the server, which will be loaded using the passed parameters. If a model checkpoint path is provided
    the client attempts to load a local model from the specified path.
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        model_checkpoint_path: Optional[Path] = None,
    ) -> None:
        super().__init__(data_path=data_path, device=device)
        self.model_checkpoint_path = model_checkpoint_path
        self.metrics = metrics
        self.local_model: Optional[nn.Module] = None
        self.global_model: Optional[nn.Module] = None
        # This data loader should be instantiated as the one on which to run evaluation
        self.data_loader: DataLoader
        self.criterion: _Loss
        self.global_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)
        self.global_metric_manager = MetricManager(self.metrics, "global_eval_manager")
        self.local_loss_meter = LossMeter.get_meter_by_type(loss_meter_type)
        self.local_metric_manager = MetricManager(self.metrics, "local_eval_manager")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        raise ValueError("Get Parameters is not impelmented for an Evaluation-Only Client")

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        raise ValueError("Fit is not implemented for an Evaluation-Only Client")

    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, parameter exchangers and other attributes for the client
        """
        self.data_loader = self.get_data_loader(config)
        self.global_model = self.initialize_global_model(config)
        self.local_model = self.get_local_model(config)

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_samples = len(self.data_loader.dataset)  # type: ignore

        self.criterion = self.get_criterion(config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)
        super().setup_client(config)

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        # Sets the global model parameters transfered from the server using a parameter exchanger to coordinate how
        # parameters are set
        if len(parameters) > 0:
            # If a non-empty set of parameters are passed, then they are inserted into a global model to be evaluated.
            # If none are provided or a global model is not instantiated, then we only evaluate a local model
            assert self.global_model is not None and self.parameter_exchanger is not None
            self.parameter_exchanger.pull_parameters(parameters, self.global_model, config)
        else:
            # If no global parameters are passed then we kill a global model (if instantiated) as it is not going to
            # be initialized with trained weights.
            self.global_model = None

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        # Make sure at least one of local or global model is not none (i.e. there is something to evaluate)
        assert self.local_model or self.global_model

        loss, metric_values = self.validate()
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_samples,
            metric_values,
        )

    def _handle_logging(self, losses: Losses, metrics_dict: Dict[str, Scalar], is_global: bool) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in losses.as_dict().items()])
        eval_prefix = "Global Model" if is_global else "Local Model"
        log(
            INFO,
            f"Client Evaluation {eval_prefix} Losses: {loss_string} \n"
            f"Client Evaluation {eval_prefix} Metrics: {metric_string}",
        )

    def validate_on_model(
        self, model: nn.Module, metric_meter: MetricManager, loss_meter: LossMeter, is_global: bool
    ) -> Tuple[Losses, Dict[str, Scalar]]:
        model.eval()
        metric_meter.clear()
        loss_meter.clear()

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                preds = model(inputs)
                losses = self.compute_loss(preds, targets)

                metric_meter.update({"predictions": preds}, targets)
                loss_meter.update(losses)

        metrics = metric_meter.compute()
        losses = loss_meter.compute()
        self._handle_logging(losses, metrics, is_global)
        return losses, metrics

    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        local_loss: Optional[Losses] = None
        local_metrics: Optional[Dict[str, Scalar]] = None

        global_loss: Optional[Losses] = None
        global_metrics: Optional[Dict[str, Scalar]] = None

        if self.local_model:
            log(INFO, "Performing evaluation on local model")
            local_loss, local_metrics = self.validate_on_model(
                self.local_model, self.local_metric_manager, self.local_loss_meter, is_global=False
            )

        if self.global_model:
            log(INFO, "Performing evaluation on global model")
            global_loss, global_metrics = self.validate_on_model(
                self.global_model, self.global_metric_manager, self.global_loss_meter, is_global=True
            )

        # Store the losses in the metrics, since we can't return more than one loss.
        metrics = EvaluateClient.merge_metrics(global_metrics, local_metrics)
        if global_loss:
            metrics.update({f"global_loss_{key}": val for key, val in global_loss.as_dict().items()})
        if local_loss:
            metrics.update({f"local_loss_{key}": val for key, val in local_loss.as_dict().items()})

        # Dummy loss is returned, global and local loss values are stored in the metrics dictionary
        return float("nan"), metrics

    @staticmethod
    def merge_metrics(
        global_metrics: Optional[Dict[str, Scalar]], local_metrics: Optional[Dict[str, Scalar]]
    ) -> Dict[str, Scalar]:
        # Merge metrics if necessary
        if global_metrics:
            metrics = global_metrics
            if local_metrics:
                for metric_name, metric_value in local_metrics.items():
                    if metric_name in metrics:
                        log(
                            WARNING,
                            f"metric_name: {metric_name} already exists in dictionary. "
                            "Please ensure that this is intended behavor",
                        )
                    metrics[metric_name] = metric_value
        elif local_metrics:
            metrics = local_metrics
        else:
            raise ValueError(
                "Both metric dictionaries are None. At least one global or local model should be present."
            )
        return metrics

    def get_parameter_exchanger(self, config: Config) -> ParameterExchanger:
        """
        Parameter exchange is assumed to always be full for evaluation only clients. If there are partial weights
        exchanged during training, we assume that the checkpoint has been saved locally. However, this functionality
        may be overridden if a different exchanger is needed
        """
        return FullParameterExchanger()

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """
        Return predictions when given input. User can override for more complex logic.
        """
        return self.model(input)

    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        """
        Computes loss given preds and torch and the user defined criterion. Optionally includes dictionary of
        loss components if you wish to train the total loss as well as sub losses if they exist.
        """
        loss = self.criterion(preds, target)
        losses = Losses(checkpoint=loss, backward=loss)
        return losses

    def get_data_loader(self, config: Config) -> DataLoader:
        """
        User defined method that returns a PyTorch DataLoader for validation
        """
        raise NotImplementedError

    def get_criterion(self, config: Config) -> _Loss:
        """
        User defined method that returns PyTorch loss to train model.
        """
        raise NotImplementedError

    def initialize_global_model(self, config: Config) -> Optional[nn.Module]:
        """
        User defined method that to initializes a global model to potentially be hydrated by parameters sent by the
        server, by default, no global model is assumed to exist unless specified by the user
        """
        return None

    def get_local_model(self, config: Config) -> Optional[nn.Module]:
        """
        Functionality for initializing a model from a local checkpoint. This can be overriden for custom behavior
        """
        # If a model checkpoint is provided, we load the checkpoint into the local model to be evaluated.
        if self.model_checkpoint_path:
            log(INFO, f"Loading model checkpoint at: {self.model_checkpoint_path.__str__()}")
            return torch.load(self.model_checkpoint_path)
        else:
            return None
