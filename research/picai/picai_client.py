from pathlib import Path
from typing import Optional, Sequence, Dict, Tuple
import torch

from flwr.common.typing import Scalar, NDArrays, Config
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric
from fl4health.checkpointing.checkpointer import TorchCheckpointer, ClientPerRoundCheckpointer
from fl4health.clients.basic_client import BasicClient

from research.picai.fl_utils import get_initial_model_ndarrays


class PicaiClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
        intermediate_checkpoint_dir: Path = Path("./")
    ) -> None:
        """
        A simple extension of the Base FL client that adds tolerance to pre-emptions by checkpointing
        each round and loading client state from checkpoint on initilization if it exists.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
                'cuda'
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to LossMeterType.AVERAGE.
            checkpointer (Optional[TorchCheckpointer], optional): Checkpointer to be used for client-side
                checkpointing. Defaults to None.
            intermediate_checkpoint_dir (Path): A directory to store and load checkpoints from for the client
                during a FL experiment.
        """
        super().__init__(data_path, metrics, device, loss_meter_type, checkpointer)
        self.per_round_checkpointer = ClientPerRoundCheckpointer(
            intermediate_checkpoint_dir, Path(f"client_{self.client_name}.pt"))

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Processes config, initializes client (if first round or restarting from pre-emption) and performs
        training based on the passed config. Overrides method in parent class to add support for client
        side checkpointing of a model thats resilient to pre-emptions. On initialization the client checks 
        if a checkpointed client state exists to load and at the end of each round the client state is saved.

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

            if self.per_round_checkpointer.checkpoint_exists():
                self.model, self.optimzers = self.per_round_checkpointer.load_checkpoint()
            else:
                self.per_round_checkpointer.save_checkpoint({
                    "model": self.model,
                    "optimizers": self.optimizers
                })

                parameters = get_initial_model_ndarrays(self.model)

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

        self.per_round_checkpointer.save_checkpoint({
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
