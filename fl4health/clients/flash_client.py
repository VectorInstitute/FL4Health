from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import torch
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.client import check_if_batch_is_empty_and_verify_input, move_data_to_device
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType


class FlashClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        """
        This client is used to perform client-side training associated with the Flash method described in
        https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf.

        Flash is designed to handle statistical heterogeneity and concept drift in federated learning through
        client-side early stopping and server-side drift-aware adaptive optimization.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        # gamma: Threshold for early stopping based on the change in validation loss. Set through the config
        self.gamma: float | None = None

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:
        """
        Performs the necessary processing of the config from the server. FLASH is not defined for step-wise training.
        So this method straps on a check to ensure that we aren't trying to do step-wise training.

        Args:
            config (Config): The config object from the server.

        Raises:
            ValueError: Throws if the user is attempting to train by steps instead of epochs for this method.

        Returns:
            (tuple[int | None, int | None, int, bool, bool]): Returns the ``local_epochs``, ``local_steps``,
                ``current_server_round``, ``evaluate_after_fit`` and ``pack_losses_with_val_metrics``. Ensures only
                one of ``local_epochs`` and ``local_steps`` is defined in the config and sets the one that is not to
                None.
        """
        local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics = (
            super().process_config(config)
        )
        if local_steps is not None:
            raise ValueError(
                "Training by steps is not applicable for FLASH clients. Please define 'local_epochs' in your"
                " config instead"
            )
        return local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics

    def train_by_epochs(
        self, epochs: int, current_round: int | None = None
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Implements a custom train_by_epochs for this client to allow for the FLASH adaptations on the client side.
        If gamma is None, then this function works exactly as the ``BasicClient``. Otherwise, we perform epochs and
        possibly stop early using gamma as a threshold.

        Args:
            epochs (int): Number of epochs to train
            current_round (int | None, optional): Current server round being performed. Defaults to None.

        Returns:
            (tuple[dict[str, float], dict[str, Scalar]]): The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        if self.gamma is None:
            return super().train_by_epochs(epochs, current_round)

        self.model.train()
        local_step = 0
        previous_loss = float("inf")
        report_data: dict = {"round": current_round}
        for local_epoch in range(epochs):
            self.train_metric_manager.clear()
            self.train_loss_meter.clear()
            self._log_header_str(current_round, local_epoch)
            report_data.update({"fit_epoch": local_epoch})
            for input, target in self.train_loader:
                if check_if_batch_is_empty_and_verify_input(input):
                    log(INFO, "Empty batch generated by data loader. Skipping step.")
                    continue

                input = move_data_to_device(input, self.device)
                target = move_data_to_device(target, self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.train_metric_manager.update(preds, target)
                self.update_after_step(local_step, current_round)
                report_data.update({"fit_losses": losses.as_dict(), "fit_step": self.total_steps})
                report_data.update(self.get_client_specific_reports())
                self.reports_manager.report(report_data, current_round, local_epoch, self.total_steps)
                self.total_steps += 1
                local_step += 1

            metrics = self.train_metric_manager.compute()
            loss_dict = self.train_loss_meter.compute().as_dict()
            current_loss, _ = self.validate()

            self._log_results(
                loss_dict,
                metrics,
                current_round=current_round,
                current_epoch=local_epoch,
            )

            if self.gamma is not None and previous_loss - current_loss < self.gamma / (local_epoch + 1):
                log(
                    INFO,
                    f"Early stopping at epoch {local_epoch} with loss change {abs(previous_loss - current_loss)}\
                        and gamma {self.gamma}",
                )
                break

            previous_loss = current_loss

        return loss_dict, metrics

    def setup_client(self, config: Config) -> None:
        """
        Follows the same flow as ``BasicClient`` for setting up the client. This function simply performs an additional
        step to process whether the gamma parameter is in the configuration.

        Args:
            config (Config): The config object from the server.
        """
        super().setup_client(config)
        if "gamma" in config:
            self.gamma = narrow_dict_type(config, "gamma", float)
        else:
            log(INFO, "Gamma not present in config. Early stopping is disabled.")
