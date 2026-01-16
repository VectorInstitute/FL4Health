import argparse
import datetime
import json
import os
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import partial
from logging import INFO
from pathlib import Path
from typing import Any

import flwr as fl
import torch
from datasets import Dataset
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from examples.fedllm_example.dataset import formatting_prompts_func, get_alpaca_tokenizer_and_data_collator, load_data
from examples.fedllm_example.model import cosine_annealing, get_model
from examples.fedllm_example.zero_utils import (
    get_peft_state_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    safe_save_model_for_zero3,
)
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.random import set_all_random_seeds


# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 2


class LlmClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        reporters: Sequence[BaseReporter],
        client_number: int,
        deepspeed_config_dir: str | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        """
        A client for finetuning a generative large language model for text generation using LoRA.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            client_number (int): The client number that uniquely identifies a client.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. For now this is disabled as we are using the HF Trainer. Defaults
                to None.
            deepspeed_config_dir (str | None, optional): The path to the deepspeed configuration file. Defaults to
                None.
            checkpoint_dir (str | None, optional): The directory to save the model checkpoints. Defaults to None.
        """
        super().__init__(data_path, metrics, device, reporters=reporters, checkpoint_and_state_module=None)
        self.client_number = client_number
        self.deepspeed_config_dir = deepspeed_config_dir

        self.training_arguments: SFTConfig
        self.trainset: Dataset
        self.train_cfg: dict[str, Any]
        self.tokenizer: PreTrainedTokenizer
        self.data_collator: Callable
        self.cosine_annealing: Callable[..., float]
        self.checkpoint_dir = checkpoint_dir

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:
        """
        This function extends the original process_config method to extract the necessary values for training
        with additional configurations for the SFTTrainer.

        Args:
            config (Config): The config from the server.

        Returns:
            (tuple[int | None, int | None, int, bool, bool]): Returns the local_epochs, local_steps,
                current_server_round, evaluate_after_fit and pack_losses_with_val_metrics. Ensures only one of
                local_epochs and local_steps is defined in the config and sets the one that is not to None.
        """
        local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics = (
            super().process_config(config)
        )
        assert isinstance(config["train"], str), "Config must contain values for train arguments."
        self.train_cfg = json.loads(config["train"])
        self.training_arguments = SFTConfig(
            **self.train_cfg.get("training_arguments", {}), deepspeed=self.deepspeed_config_dir
        )

        # Set the maximum number of steps to `local_steps`
        self.training_arguments.max_steps = local_steps

        # Set the per_device_train_batch_size and per_device_eval_batch_size in training_arguments based on
        # client's specified `batch_size`. This configuration results in training the model with a total
        # `batch_size` for `local_steps` number of iterations.
        batch_size = config.get("batch_size")
        num_gpus_per_client = config.get("num_gpus_per_client")

        assert isinstance(batch_size, int) and isinstance(num_gpus_per_client, int)
        assert batch_size % num_gpus_per_client == 0, "Batch size must be divisible by number of GPUs per client"
        self.training_arguments.per_device_train_batch_size = batch_size // num_gpus_per_client
        self.training_arguments.per_device_eval_batch_size = batch_size // num_gpus_per_client

        # Set the output directory for the model
        self.training_arguments.output_dir = self.checkpoint_dir

        # Set the dtype for the model
        self.compute_dtype = (
            torch.float16
            if self.training_arguments.fp16
            else (torch.bfloat16 if self.training_arguments.bf16 else torch.float32)
        )

        # Set the learning rate scheduler for each iteration
        self.cosine_annealing = partial(cosine_annealing, total_round=int(config["n_server_rounds"]))

        # Either local epochs or local steps is none based on what key is passed in the config
        return local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics

    def set_parameters(self, parameters: NDArrays, config: Config, fitting_round: bool) -> None:
        # We have to reset model on each new round as deepspeed would crash due to double initialization and
        # partitioning of the model
        self.model = get_model(self.model_cfg)

        peft_state_dict_keys = get_peft_model_state_dict(self.model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.model, state_dict)

    def get_parameters(self, config: Config) -> NDArrays:
        """Return the parameters of the current net."""
        # In deepspeed Stage 3, we need to get lora parameters differently as all the parameters are also partitioned.
        # We should make sure to gather all of these parameters for sending them to server.
        if not self.deepspeed_config_dir or (self.deepspeed_config_dir and "zero3" not in self.deepspeed_config_dir):
            state_dict = get_peft_model_state_dict(self.model)
        else:
            state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), "none")
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_train_dataset(self, config: Config) -> None:
        """
        Set the train dataset for the client using the partition id and the number of partitions.

        Args:
            config (Config): The config from the server.
        """
        # Let's get the client partition
        assert isinstance(config["dataset"], str), "Config must contain values for dataset arguments."
        dataset_cfg = json.loads(config["dataset"])
        assert isinstance(dataset_cfg, dict), "Dataset configuration must be a dictionary"
        dataset = load_data(self.client_number, NUM_CLIENTS, dataset_cfg["name"])

        # Split the dataset into train and validation
        split_dataset = dataset.train_test_split(test_size=0.1)

        self.client_trainset = split_dataset["train"]
        self.client_testset = split_dataset["test"]

    def get_model(self, config: Config) -> nn.Module:
        """
        Get the model based on the configuration provided by the server. Also load respective tokenizer and data
        collator.

        Args:
            config (Config): The config from the server.
        """
        assert isinstance(config["model"], str), "Config must contain values for model arguments."
        self.model_cfg = json.loads(config["model"])
        assert isinstance(self.model_cfg, dict), "Model configuration must be a dictionary"
        (
            self.tokenizer,
            self.data_collator,
        ) = get_alpaca_tokenizer_and_data_collator(self.model_cfg["name"])

        assert isinstance(self.model_cfg, dict), "Model configuration must be a dictionary"
        assert self.model_cfg.get("gradient_checkpointing", False) == self.train_cfg["training_arguments"].get(
            "gradient_checkpointing", False
        )
        return get_model(self.model_cfg)

    def setup_client(self, config: Config) -> None:
        """
        Override the ``setup_client`` method to set the model, train dataset, and other necessary configurations.

        Args:
            config (Config): The config from the server.
        """
        self.model = self.get_model(config)

        self.set_train_dataset(config)

        self.num_train_samples = len(self.client_trainset)
        self.num_val_samples = len(self.client_testset)

        self.reports_manager.report({"host_type": "client", "initialized": str(datetime.datetime.now())})
        self.initialized = True

    def update_before_train(self, current_server_round: int) -> None:
        """
        Hook method called before training with the number of current server rounds performed.
        In this method we update the learning rate for the current round and set the training arguments
        for the trainer. We also construct the trainer object.

        Args:
            current_server_round (int): The number of current server round.
        """
        assert isinstance(self.train_cfg, dict), "train_cfg should be a dictionary"
        lrate_max = self.train_cfg.get("learning_rate_max")
        lrate_min = self.train_cfg.get("learning_rate_min")
        assert lrate_max is not None, "learning_rate_max is missing in train_cfg"
        assert lrate_min is not None, "learning_rate_min is missing in train_cfg"

        self.lr = self.cosine_annealing(
            current_round=current_server_round,
            learning_rate_max=float(lrate_max),
            learning_rate_min=float(lrate_min),
        )

        assert isinstance(self.training_arguments, SFTConfig), (
            "training_arguments should be a TrainingArguments object"
        )

        self.training_arguments.learning_rate = self.lr
        # Disable reporting to avoid cluttering the logs
        self.training_arguments.report_to = "none"

        # Construct trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=self.client_trainset,
            eval_dataset=self.client_testset,
            formatting_func=formatting_prompts_func,
            data_collator=self.data_collator,
        )

    def train_by_steps(
        self,
        steps: int,
        current_round: int | None = None,
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.
            current_round (int | None, optional): The current FL round

        Returns:
            (tuple[dict[str, float], dict[str, Scalar]]): The loss and metrics dictionary from the local training.
                Loss is a dictionary of one or more losses that represent the different components of the loss.
        """
        # Do local training
        results = self.trainer.train()
        loss_dict = {"train_loss": results.training_loss}
        # Pass the training loss to the metrics, further metrics can be implemented via evaluate and
        # get added to here
        metrics = {"train_loss": results.training_loss}

        return loss_dict, metrics

    def update_after_train(self, local_steps: int, loss_dict: dict[str, float], config: Config) -> None:
        """
        We use this function to save the model after training. This can later be used to have a checkpointing
        mechanism for the client based on the loss_dict.

        Args:
            local_steps (int): The number of local steps trained.
            loss_dict (dict[str, float]): The loss dictionary from the local training.
            config (Config): The config from the server.
        """
        self.trainer.save_state()
        # type ignore because the type in transformers is very indirect
        self.model.config.use_cache = True  # type: ignore

        # In deepspeed Stage 3, we need to save the model differently as all the parameters are also partitioned. We
        # should make sure to gather all of these parameters before saving the model, for safe loading and resuming.
        if not self.deepspeed_config_dir or (self.deepspeed_config_dir and "zero3" not in self.deepspeed_config_dir):
            safe_save_model_for_hf_trainer(trainer=self.trainer)
        else:
            safe_save_model_for_zero3(model=self.model, training_arguments=self.training_arguments)

        return super().update_after_train(local_steps, loss_dict, config)

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation set.

        Returns:
            (tuple[float, dict[str, Scalar]]): The validation loss and a dictionary of metrics from validation (and
                test if present).
        """
        results = self.trainer.evaluate()

        val_loss = results.get("eval_loss")
        val_metrics = {"val_loss": val_loss}

        return val_loss, val_metrics


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
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--deepspeed",
        action="store",
        type=str,
        help="Path to the deepspeed configuration file",
        required=False,
        default=None,
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    # Set the random seed for reproducibility
    set_all_random_seeds(args.seed)

    # Set the checkpoint directory
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)

    client = LlmClient(
        data_path=Path(" "),
        metrics=[Accuracy()],
        device=device,
        reporters=[JsonReporter()],
        client_number=args.client_number,
        deepspeed_config_dir=args.deepspeed,
        checkpoint_dir=checkpoint_dir,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
