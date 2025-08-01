import datetime
from collections.abc import Sequence
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.metrics.metric_managers import MetricManager
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.utils.losses import EvaluationLosses, LossMeter, LossMeterType
from fl4health.utils.random import generate_hash


class EvaluateClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        model_checkpoint_path: Path | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        client_name: str | None = None,
    ) -> None:
        """
        This client implements an evaluation only flow. That is, there is no expectation of parameter exchange with
        the server past the model initialization stage. The implementing client should instantiate a global model if
        one is expected from the server, which will be loaded using the passed parameters. If a model checkpoint path
        is provided the client attempts to load a local model from the specified path.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            model_checkpoint_path (Path | None, optional): Path to which the model should be checkpointed. Defaults to
                None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Defaults to None.
        """
        # EvaluateClient does not call BasicClient constructor and sets attributes
        # in a custom way to account for the fact it does not involve any training
        self.client_name = generate_hash() if client_name is None else client_name
        self.data_path = data_path
        self.device = device
        self.model_checkpoint_path = model_checkpoint_path
        self.metrics = metrics
        self.initialized = False

        # Initialize reporters with client information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.client_name)

        # This data loader should be instantiated as the one on which to run evaluation
        self.global_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.global_metric_manager = MetricManager(self.metrics, "global_eval_manager")
        self.local_loss_meter = LossMeter[EvaluationLosses](loss_meter_type, EvaluationLosses)
        self.local_metric_manager = MetricManager(self.metrics, "local_eval_manager")

        # The attributes to be set in setup_client
        # Models corresponding to client-side and server-side checkpoints,
        # if they exist, to be evaluated on the client's dataset.
        self.data_loader: DataLoader
        self.criterion: _Loss
        self.local_model: nn.Module | None = None
        self.global_model: nn.Module | None = None

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        raise ValueError("Get Parameters is not implemented for an Evaluation-Only Client")

    def fit(self, parameters: NDArrays, config: Config) -> tuple[NDArrays, int, dict[str, Scalar]]:
        raise ValueError("Fit is not implemented for an Evaluation-Only Client")

    def setup_client(self, config: Config) -> None:
        """Set dataloaders, parameter exchangers and other attributes for the client."""
        (data_loader,) = self.get_data_loader(config)
        self.data_loader = data_loader
        self.global_model = self.initialize_global_model(config)
        self.local_model = self.get_local_model(config)

        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_samples = len(self.data_loader.dataset)  # type: ignore

        self.criterion = self.get_criterion(config)
        self.parameter_exchanger = self.get_parameter_exchanger(config)

        self.reports_manager.report({"host_type": "client", "initialized": str(datetime.datetime.now())})

        self.initialized = True

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        assert not fitting_round
        # Sets the global model parameters transferred from the server using a parameter exchanger to coordinate how
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

    def evaluate(self, parameters: NDArrays, config: Config) -> tuple[float, int, dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config)

        start_time = datetime.datetime.now()
        self.set_parameters(parameters, config, fitting_round=False)
        # Make sure at least one of local or global model is not none (i.e. there is something to evaluate)
        assert self.local_model or self.global_model

        loss, metric_values = self.validate()
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time

        self.reports_manager.report(
            {
                "eval_metrics": metric_values,
                "eval_loss": loss,
                "eval_start": str(start_time),
                "eval_time_elapsed": str(elapsed),
                "eval_end": str(end_time),
            },
            0,
        )

        # EvaluateRes should return the loss, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            loss,
            self.num_samples,
            metric_values,
        )

    def _handle_logging(  # type: ignore
        self, losses: EvaluationLosses, metrics_dict: dict[str, Scalar], is_global: bool
    ) -> None:
        metric_string = "\t".join([f"{key}: {str(val)}" for key, val in metrics_dict.items()])
        loss_string = "\t".join([f"{key}: {str(val)}" for key, val in losses.as_dict().items()])
        eval_prefix = "Global Model" if is_global else "Local Model"
        log(
            INFO,
            f"Client Evaluation {eval_prefix} Losses: {loss_string} \n"
            f"Client Evaluation {eval_prefix} Metrics: {metric_string}",
        )

    def validate_on_model(
        self,
        model: nn.Module,
        metric_meter: MetricManager,
        loss_meter: LossMeter,
        is_global: bool,
    ) -> tuple[EvaluationLosses, dict[str, Scalar]]:
        model.eval()
        metric_meter.clear()
        loss_meter.clear()
        model.to(self.device)

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                preds = {"prediction": model(inputs)}
                losses = self.compute_evaluation_loss(preds, {}, targets)

                metric_meter.update(preds, targets)
                loss_meter.update(losses)

        metrics = metric_meter.compute()
        losses = loss_meter.compute()
        self._handle_logging(losses, metrics, is_global)
        return losses, metrics

    def validate(self, include_loss_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        local_loss: EvaluationLosses | None = None
        local_metrics: dict[str, Scalar] | None = None

        global_loss: EvaluationLosses | None = None
        global_metrics: dict[str, Scalar] | None = None

        if self.local_model:
            log(INFO, "Performing evaluation on local model")
            local_loss, local_metrics = self.validate_on_model(
                self.local_model,
                self.local_metric_manager,
                self.local_loss_meter,
                is_global=False,
            )

        if self.global_model:
            log(INFO, "Performing evaluation on global model")
            global_loss, global_metrics = self.validate_on_model(
                self.global_model,
                self.global_metric_manager,
                self.global_loss_meter,
                is_global=True,
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
        global_metrics: dict[str, Scalar] | None,
        local_metrics: dict[str, Scalar] | None,
    ) -> dict[str, Scalar]:
        # Merge metrics if necessary
        if global_metrics:
            metrics = global_metrics
            if local_metrics:
                for metric_name, metric_value in local_metrics.items():
                    if metric_name in metrics:
                        log(
                            WARNING,
                            f"metric_name: {metric_name} already exists in dictionary. "
                            "Please ensure that this is intended behavior",
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
        may be overridden if a different exchanger is needed.
        """
        return FullParameterExchanger()

    def get_data_loader(self, config: Config) -> tuple[DataLoader]:
        """User defined method that returns a PyTorch DataLoader for validation."""
        raise NotImplementedError

    def initialize_global_model(self, config: Config) -> nn.Module | None:
        """
        User defined method that to initializes a global model to potentially be hydrated by parameters sent by the
        server, by default, no global model is assumed to exist unless specified by the user.
        """
        return None

    def get_local_model(self, config: Config) -> nn.Module | None:
        """
        Functionality for initializing a model from a local checkpoint. This can be overridden for custom
        behavior.
        """
        # If a model checkpoint is provided, we load the checkpoint into the local model to be evaluated.
        if self.model_checkpoint_path:
            log(
                INFO,
                f"Loading model checkpoint at: {str(self.model_checkpoint_path)}",
            )
            return torch.load(self.model_checkpoint_path)
        return None
