from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple
import torch

from flwr.common.typing import Scalar, NDArrays, Config
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.checkpointing.checkpointer import TorchCheckpointer, ClientPerEpochCheckpointer
from fl4health.clients.basic_client import BasicClient


class PicaiClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        per_epoch_checkpointer: ClientPerEpochCheckpointer = ClientPerEpochCheckpointer(
            checkpoint_dir="./", checkpoint_name="ckpt.pt")
    ) -> None:
        super().__init__(data_path, metrics, device, loss_meter_type, checkpointer)
        self.per_epoch_checkpointer = per_epoch_checkpointer

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round) and performs training based on the passed config.

        Args:
            parameters (NDArrays): The parameters of the model to be used in fit.
            config (NDArrays): The config from the server.

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: The parameters following the local training along with the
            number of samples in the local training dataset and the computed metrics throughout the fit.

        Raises:
            ValueError: If local_steps or local_epochs is not specified in config.
        """
        local_epochs, local_steps, current_server_round = self.process_config(config)

        if not self.initialized:
            self.setup_client(config)

            if self.per_epoch_checkpointer.checkpoint_exists():
                self.model, self.optimzers = self.per_epoch_checkpointer.load_checkpoint()
            else:
                self.per_epoch_checkpointer.save_checkpoint({
                    "model": self.model,
                    "optimizers": self.optimizers
                })

        self.set_parameters(parameters, config)

        if local_epochs is not None:
            loss_dict, metrics = self.train_by_epochs(local_epochs, current_server_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
        elif local_steps is not None:
            loss_dict, metrics = self.train_by_steps(local_steps, current_server_round)
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, loss_dict)

        self.per_epoch_checkpointer.save_checkpoint({
            "model": self.model,
            "optimizers": self.optimizers
        })

        # FitRes should contain local parameters, number of examples on client, and a dictionary holding metrics
        # calculation results.
        return (
            self.get_parameters(config),
            self.num_train_samples,
            metrics,
        )
