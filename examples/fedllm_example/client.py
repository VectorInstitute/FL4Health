import argparse
import datetime
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
import torch.nn as nn
from datasets import Dataset
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer  # type: ignore

from examples.fedllm_example.dataset import formatting_prompts_func, get_tokenizer_and_data_collator, load_data
from examples.fedllm_example.model import cosine_annealing, get_model
from examples.fedllm_example.zero_utils import safe_save_model_for_hf_trainer, safe_save_model_for_zero3
from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.reporting import JsonReporter
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.metrics import Accuracy, Metric
from fl4health.utils.random import set_all_random_seeds

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = 2


class LLMClient(BasicClient):
    """A client for training a generative large language model."""

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        reporters: Sequence[BaseReporter],
        client_number: int,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        deepspeed_config: str | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        super().__init__(
            data_path, metrics, device, reporters=reporters, checkpoint_and_state_module=checkpoint_and_state_module
        )
        self.client_number = client_number
        self.deepspeed_config = deepspeed_config

        self.training_arguments: TrainingArguments
        self.trainset: Dataset
        self.train_cfg: dict[str, Any]
        self.tokenizer: PreTrainedTokenizer
        self.data_collator: Callable
        self.cosine_annealing: Callable[..., float]
        self.checkpoint_dir = checkpoint_dir

    def get_unflatten_config(self, flat_dict: dict[str, Scalar], prefix: str = "train") -> dict[str, Any]:
        """
        extract the sub-dictionary with the given prefix and unflatten the config resulting in a nested
        dictionary. The flat dictionary is expected to have keys with the format 'prefix#key1#key2#...#keyN'
        where the keys are split by '#'. The unflattened dictionary will have the prefix as the root key and
        the sub-dictionary as the value.

        Args:
            flat_dict (dict[str, Scalar]): The flat dictionary to unflatten.
            prefix (str, optional): The prefix to use for the unflattened dictionary. Defaults to "train".

        Returns:
            dict[str, Any]: The unflattened dictionary with the given prefix.
        """
        dictionary: dict[str, Any] = {}

        for key, value in flat_dict.items():
            if key.startswith(f"{prefix}#"):
                keys = key[len(prefix) + 1 :].split("#")  # Remove 'train_' prefix and split
                d = dictionary.setdefault(prefix, {})  # Ensure 'train' is the root key
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value

        return dictionary

    def process_config(self, config: Config) -> tuple[int | None, int | None, int, bool, bool]:
        """
        This function extend the original process_config method to extract the necessary values for training
        with additional configurations for the SFTTrainer.

        Args:
            config (Config): The config from the server.

        Returns:
            tuple[int | None, int | None, int, bool, bool]: Returns the local_epochs, local_steps,
                current_server_round, evaluate_after_fit and pack_losses_with_val_metrics. Ensures only one of
                local_epochs and local_steps is defined in the config and sets the one that is not to None.
        """

        local_epochs, local_steps, current_server_round, evaluate_after_fit, pack_losses_with_val_metrics = (
            super().process_config(config)
        )
        self.train_cfg = self.get_unflatten_config(config, "train")["train"]
        assert isinstance(self.train_cfg, dict), "Config must contain values for train arguments."
        self.training_arguments = TrainingArguments(
            **self.train_cfg.get("training_arguments", {}), deepspeed=self.deepspeed_config
        )
        # We will set the number of max_steps to the local_steps
        self.training_arguments.max_steps = local_steps
        self.training_arguments.per_device_train_batch_size = config.get("batch_size")
        self.training_arguments.output_dir = self.checkpoint_dir

        log(INFO, f"Device local rank is {self.training_arguments.local_rank}")

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
        log(INFO, "Setting parameters")
        assert self.model is not None

        peft_state_dict_keys = get_peft_model_state_dict(self.model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.model, state_dict)

    def get_parameters(self, config: Config) -> NDArrays:
        """Return the parameters of the current net."""
        log(INFO, "Getting parameters")
        state_dict = get_peft_model_state_dict(self.model)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_train_dataset(self, config: Config) -> None:
        """
        Set the train dataset for the client using the partition id and the number of partitions.

        Args:
            config (Config): The config from the server.
        """
        partition_id = self.client_number
        num_partitions = NUM_CLIENTS

        # Let's get the client partition
        dataset_cfg = self.get_unflatten_config(config, "dataset")["dataset"]
        assert isinstance(dataset_cfg, dict), "Dataset configuration must be a dictionary"
        dataset = load_data(partition_id, num_partitions, dataset_cfg["name"])

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

        model_cfg = self.get_unflatten_config(config, "model")["model"]
        assert isinstance(model_cfg, dict), "Model configuration must be a dictionary"
        (
            self.tokenizer,
            self.data_collator,
        ) = get_tokenizer_and_data_collator(model_cfg["name"])

        assert isinstance(model_cfg, dict), "Model configuration must be a dictionary"
        return get_model(model_cfg)

    def setup_client(self, config: Config) -> None:
        """
        Override the setup_client method to set the model, train dataset, and other necessary configurations.

        Args:
            config (Config): The config from the server.
        """
        log(INFO, "Setting up client")

        self.model = self.get_model(config).to(self.device)

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
            lrate_max=float(lrate_max),
            lrate_min=float(lrate_min),
        )

        assert isinstance(
            self.training_arguments, TrainingArguments
        ), "training_arguments should be a TrainingArguments object"

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
        epochs: int,
        current_round: int | None = None,
    ) -> tuple[dict[str, float], dict[str, Scalar]]:
        """
        Train locally for the specified number of steps.

        Args:
            steps (int): The number of steps to train locally.
            current_round (int | None, optional): The current FL round

        Returns:
            tuple[dict[str, float], dict[str, Scalar]]: The loss and metrics dictionary from the local training.
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
        self.model.config.use_cache = True

        if not self.deepspeed_config or (self.deepspeed_config and "zero3" not in self.deepspeed_config):
            safe_save_model_for_hf_trainer(trainer=self.trainer, output_dir=self.training_arguments.output_dir)
        else:
            safe_save_model_for_zero3(self.model, self.training_arguments)

        return super().update_after_train(local_steps, loss_dict, config)

    def validate(self, include_losses_in_metrics: bool = False) -> tuple[float, dict[str, Scalar]]:
        """
        Validate the current model on the entire validation set.

        Returns:
            tuple[float, dict[str, Scalar]]: The validation loss and a dictionary of metrics
            from validation (and test if present).
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

    # Adding extensive checkpointing for the client
    checkpoint_dir = os.path.join(args.artifact_dir, args.run_name)

    client = LLMClient(
        data_path=Path(" "),
        metrics=[Accuracy()],
        device=device,
        reporters=[JsonReporter()],
        client_number=args.client_number,
        deepspeed_config=args.deepspeed,
        checkpoint_dir=checkpoint_dir,
    )

    fl.client.start_client(server_address=args.server_address, client=client.to_client())
