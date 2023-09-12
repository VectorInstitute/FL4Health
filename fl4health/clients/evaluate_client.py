from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.utils.metrics import AverageMeter, Meter, Metric

T = TypeVar("T")


class EvaluateClient(NumPyClient):
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
        model_checkpoint_path: Optional[Path] = None,
    ) -> None:
        self.data_path = data_path
        self.device = device
        self.model_checkpoint_path = model_checkpoint_path
        self.metrics = metrics
        self.local_model: Optional[nn.Module] = None
        self.global_model: Optional[nn.Module] = None
        # This data loader should be instantiated as the one on which to run evaluation
        self.data_loader: DataLoader
        self.num_examples: Dict[str, int]
        self.criterion: _Loss

        self.initialized = False
        # Parameter exchange is assumed to always be full for evaluation only clients. If there are partial weights
        # exchanged during training, we assume that the checkpoint has been saved locally.
        self.parameter_exchanger = FullParameterExchanger()
        # If a model checkpoint is provided, we load the checkpoint into the local model to be evaluated.
        if model_checkpoint_path:
            self.load_local_model_checkpoint()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        raise ValueError("Get Parameters is not impelmented for an Evaluation-Only Client")

    def load_local_model_checkpoint(self) -> None:
        assert self.model_checkpoint_path is not None
        log(INFO, f"Loading model checkpoint at: {self.model_checkpoint_path.__str__()}")
        self.local_model = torch.load(self.model_checkpoint_path)

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

    def setup_client(self, config: Config) -> None:
        """
        This method should be used to set up all of the required components for the client through the config passed
        by the server and need only be done once. The quintessential example is data loaders with a batch size set by
        the server in the config. The parameter initialized should be set to true when this function is finished.
        Overriding this class and calling super is the preferred flow.
        """
        self.initialized = True

    def narrow_config_type(self, config: Config, config_key: str, narrow_type_to: Type[T]) -> T:
        config_value = config[config_key]
        if isinstance(config_value, narrow_type_to):
            return config_value
        else:
            raise ValueError(f"Provided configuration key ({config_key}) value does not have correct type")

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        raise ValueError("Fit is not implemented for an Evaluation-Only Client")

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        self.set_parameters(parameters, config)
        # Make sure at least one of local or global model is not none (i.e. there is something to evaluate)
        assert self.local_model or self.global_model
        global_meter = AverageMeter(self.metrics, "global_eval_meter") if self.global_model else None
        local_meter = AverageMeter(self.metrics, "local_eval_meter") if self.local_model else None

        loss, metric_values = self.validate(global_meter, local_meter)
        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_examples["eval_set"],
            metric_values,
        )

    def _handle_logging(self, loss: float, metrics_dict: Dict[str, Scalar], is_global: bool) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        eval_prefix = "Global Model" if is_global else "Local Model"
        log(
            INFO,
            f"Client Evaluation {eval_prefix} Loss: {loss} \n"
            f"Client Evaluation {eval_prefix} Metrics: {metric_string}",
        )

    def validate_on_model(self, model: nn.Module, meter: Meter, is_global: bool) -> Tuple[float, Dict[str, Scalar]]:
        model.eval()
        meter.clear()
        running_loss = 0.0

        with torch.no_grad():
            for input, target in self.data_loader:
                input, target = input.to(self.device), target.to(self.device)
                pred = model(input)
                loss = self.criterion(pred, target)

                running_loss += loss.item()
                meter.update(pred, target)

        running_loss = running_loss / len(self.data_loader)
        metrics = meter.compute()
        self._handle_logging(running_loss, metrics, is_global)
        return running_loss, metrics

    def validate(self, global_meter: Optional[Meter], local_meter: Optional[Meter]) -> Tuple[float, Dict[str, Scalar]]:
        local_loss: Optional[float] = None
        local_metrics: Optional[Dict[str, Scalar]] = None

        global_loss: Optional[float] = None
        global_metrics: Optional[Dict[str, Scalar]] = None

        if self.local_model and local_meter:
            log(INFO, "Performing evaluation on local model")
            local_loss, local_metrics = self.validate_on_model(self.local_model, local_meter, is_global=False)

        if self.global_model and global_meter:
            log(INFO, "Performing evaluation on global model")
            global_loss, global_metrics = self.validate_on_model(self.global_model, global_meter, is_global=True)

        # Store the losses in the metrics, since we can't return more than one loss.
        metrics = EvaluateClient.merge_metrics(global_metrics, local_metrics)
        if global_loss:
            metrics["global_loss"] = global_loss
        if local_loss:
            metrics["local_loss"] = local_loss

        return 0.0, metrics

    @staticmethod
    def merge_metrics(
        global_metrics: Optional[Dict[str, Scalar]], local_metrics: Optional[Dict[str, Scalar]]
    ) -> Dict[str, Scalar]:
        # Merge metrics if necessary
        if global_metrics:
            metrics = global_metrics
            if local_metrics:
                for metric_name, metric_value in local_metrics.items():
                    metrics[metric_name] = metric_value
        elif local_metrics:
            metrics = local_metrics
        else:
            raise ValueError(
                "Both metric dictionaries are None. At least one global or local model should be present, but is not"
            )
        return metrics
