import argparse
import datetime
from logging import INFO
from pathlib import Path
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import Config
from datasets import Dataset

from fl4health.checkpointing.checkpointer import LatestTorchModuleCheckpointer
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.utils.random import set_all_random_seeds
from fl4health.clients.basic_client import BasicClient
from fl4health.reporting.base_reporter import BaseReporter

import os
import warnings
from typing import Dict

import torch
from flwr.common.typing import NDArrays, Scalar

from transformers import TrainingArguments, PreTrainedTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM  # type: ignore

from examples.fedllm_example.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
)
from examples.fedllm_example.model import (
    cosine_annealing,
    get_model,
)


from collections import OrderedDict
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from fl4health.utils.metrics import Accuracy
from fl4health.reporting import JsonReporter
from fl4health.utils.metrics import Metric

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 4
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class LLMClient(BasicClient):
    """Standard Flower client for CNN training."""
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        reporters: Sequence[BaseReporter],
        client_number: int,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
    ) -> None:
        super().__init__(data_path, metrics, device, reporters = reporters,
                         checkpoint_and_state_module = checkpoint_and_state_module)
        self.client_number = client_number
        self.training_arguments: TrainingArguments 
        self.trainset: Dataset
        self.train_cfg: dict[str, Any]
        self.cosine_annealing: Callable[[int, float, float], int]
        self.tokenizer: PreTrainedTokenizer
        self.formatting_prompts_func: DataCollatorForCompletionOnlyLM
        self.data_collator: Callable

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:

    
        local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics = super().process_config(config)
        train_cfg = config.get("train")
        if not isinstance(train_cfg, dict):
            raise TypeError("Config must contain a 'train' key with a dictionary value.")
        self.train_cfg = train_cfg
        assert isinstance(self.train_cfg, dict), "Config must contain a 'train' key with a dictionary value."
        self.training_arguments = TrainingArguments(**self.train_cfg.get("training_arguments", {}))
        self.training_arguments.per_device_train_batch_size = config.get("batch_size")

        self.cosine_annealing = partial(cosine_annealing, total_round= config["n_server_rounds"])
        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics
    
    def update_before_train(self, current_server_round: int) -> None:
        """
        Hook method called before training with the number of current server rounds performed.
        NOTE: This method is called immediately AFTER the aggregated parameters are received from the server.
        For example, used by MOON and FENDA to save global modules after aggregation.

        Args:
            current_server_round (int): The number of current server round.
        """
        assert isinstance(self.train_cfg, dict), "train_cfg should be a dictionary"
        lrate_max = self.train_cfg.get("learning_rate_max")
        lrate_min = self.train_cfg.get("learning_rate_min")
        assert lrate_max is not None, "learning_rate_max is missing in train_cfg"
        assert lrate_min is not None, "learning_rate_min is missing in train_cfg"

        self.lr = self.cosine_annealing(
            current_server_round,
            lrate_max,
            lrate_min,
        )
        self.training_arguments["learning_rate"] = self.lr 

    def train_by_epochs(
        self,
        epochs: int,
        current_round: int | None = None,
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        
        assert isinstance(self.training_arguments, TrainingArguments), "training_arguments should be a TrainingArguments object"
        
        self.training_arguments.num_train_epochs = epochs
        

        ### train by epoch
        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.get("seq_length"),
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()
        loss_dict = {"train_loss": results.training_loss}
        metrics = {"train_loss": results.training_loss}

    

        # Return final training metrics
        return loss_dict, metrics

    
    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        assert self.model is not None

        peft_state_dict_keys = get_peft_model_state_dict(self.model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.model, state_dict)
        
    
    def get_parameters(self, config: Config) -> NDArrays:
        """Return the parameters of the current net."""
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]
    
    def set_train_dataset(self, config: Config)-> None:
        ### Should be impelement
        """
        User defined method that returns a PyTorch Train DataLoader
        and a PyTorch Validation DataLoader

        Args:
            config (Config): The config from the server.

        Returns:
            tuple[DataLoader, ...]: Tuple of length 2. The client train and validation loader.

        Raises:
            NotImplementedError: To be defined in child class.
        """
        partition_id = self.client_number
        num_partitions = NUM_CLIENTS

        # Let's get the client partition
        dataset_cfg= config.get("dataset")
        assert isinstance(dataset_cfg, dict), "Dataset configuration must be a dictionary"
        self.client_trainset = load_data(partition_id, num_partitions, dataset_cfg["name"])
        (
            self.tokenizer,
            self.data_collator,
            self.formatting_prompts_func,
        ) = get_tokenizer_and_data_collator_and_propt_formatting(config["model"]["name"])

    
    def get_model(self, config: Config) -> nn.Module:
        ### Done
        model_cfg = config.get("model")
        assert isinstance(model_cfg, dict), "Model configuration must be a dictionary"
        return get_model(model_cfg)
    
    def setup_client(self, config: Config) -> None:
        """
        Set dataloaders, optimizers, parameter exchangers and other attributes derived from these.
        Then set initialized attribute to True.

        Args:
            config (Config): The config from the server.
        """
        # Explicitly send the model to the desired device. This is idempotent.
        self.model = self.get_model(config).to(self.device)
        
        self.set_train_dataset(config)
        # The following lines are type ignored because torch datasets are not "Sized"
        # IE __len__ is considered optionally defined. In practice, it is almost always defined
        # and as such, we will make that assumption.
        self.num_train_samples = len(self.client_trainset)  # type: ignore

        self.reports_manager.report({"host_type": "client", "initialized": str(datetime.datetime.now())})
        self.initialized = True





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    parser.add_argument(
        "--client_number",
        type=int,
        help="The partition id of the client",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        help="Seed for the random number generators across python, torch, and numpy",
        required=False,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)
    pre_aggregation_last_checkpoint_name = f"pre_aggregation_client_{args.client_number}_last_model.pkl"
    checkpoint_and_state_module = ClientCheckpointAndStateModule(
        pre_aggregation=[
            LatestTorchModuleCheckpointer(checkpoint_dir, pre_aggregation_last_checkpoint_name),
        ],
    )

    client = LLMClient(data_path=Path(" "), 
                        metrics=[Accuracy()],
                        device=device, 
                        reporters=[JsonReporter()], 
                        client_number=args.client_number, 
                        checkpoint_and_state_module=checkpoint_and_state_module)

    fl.client.start_client(server_address=args.server_address, client=client.to_client())