from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from fl4health.clients.basic_client import BasicClient

from fl4health.checkpointing.state_checkpointer import ClientStateCheckpointer
from fl4health.utils.logging import LoggingMode


class EarlyStopper:
    def __init__(
        self,
        client: BasicClient,
        train_loop_checkpoint_dir: Path,
        patience: int | None = 1,
        interval_steps: int = 5,
    ) -> None:
        """
        Early stopping class is a plugin for the client that allows to stop local training based on the validation
        loss. At each training step this class saves the best state of the client and restores it if the client is
        stopped. If the client starts to overfit, the early stopper will stop the training process and restore the best
        state of the client before sending the model to the server.

        Args:
            client (BasicClient): The client to be monitored.
            train_loop_checkpoint_dir (Path): Directory to checkpoint the "best" state seen so far.
            patience (int, optional): Number of validation cycles to wait before stopping the training. If it is equal
                to None client never stops, but still loads the best state before sending the model to the server.
                Defaults to 1.
            interval_steps (int): Specifies the frequency, in terms of training intervals, at which the early
                stopping mechanism should evaluate the validation loss. Defaults to 5.
        """
        self.client = client

        self.patience = patience
        self.count_down = patience
        self.interval_steps = interval_steps

        # Early stopper uses a default name for the state
        checkpoint_name = f"temp_{self.client.client_name}.pt"

        self.state_checkpointer = ClientStateCheckpointer(
            checkpoint_dir=train_loop_checkpoint_dir, checkpoint_name=checkpoint_name
        )

        self.best_score: float | None = None

    def load_snapshot(self, attributes: list[str] | None = None) -> None:
        """
        Load the best snapshot of the client state from the checkpoint directory.

        Args:
            attributes (list[str] | None, optional): List of attributes to load from the checkpoint.
                If None, all attributes as defined in ``state_checkpointer`` are loaded. Defaults to None.
        """
        # Load the best snapshot, and update self.client with the values
        self.state_checkpointer.maybe_load_client_state(self.client, attributes)

    def should_stop(self, steps: int) -> bool:
        """
        Determine if the client should stop training based on early stopping criteria.

        Args:
            steps (int): Number of steps since the start of the training.

        Returns:
            (bool): True if training should stop, otherwise False.
        """
        if steps % self.interval_steps != 0:
            return False

        val_loss, _ = self.client._fully_validate_or_test(
            loader=self.client.val_loader,
            loss_meter=self.client.val_loss_meter,
            metric_manager=self.client.val_metric_manager,
            logging_mode=LoggingMode.EARLY_STOP_VALIDATION,
            include_losses_in_metrics=False,
        )

        if val_loss is None:
            return False

        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            self.count_down = self.patience
            self.state_checkpointer.save_client_state(self.client)
            return False

        if self.count_down is not None:
            self.count_down -= 1
            if self.count_down <= 0:
                return True

        return False
