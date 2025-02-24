import os
from collections.abc import Iterator
from logging import WARNING
from typing import Any

import torch
import transformers
from deepspeed import zero  # type: ignore
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # type: ignore
from flwr.common.logger import log
from torch.nn import Parameter


def maybe_zero_3(param: Any, ignore_status: bool = False, name: str | None = None) -> Any:

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                log(WARNING, f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params: Iterator[tuple[str, Parameter]], bias: str) -> dict[str, Any]:
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(
    named_params: Iterator[tuple[str, Parameter]], require_grad_only: bool = True
) -> dict[str, Any]:
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_zero3(model: torch.nn.Module, training_arguments: transformers.TrainingArguments) -> None:
    """
    Saves PEFT model and non-LoRA trainable parameters.

    Args:
        model: The model to save.
        training_arguments: Training arguments containing local_rank and output directory.
    """
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), "none")
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())

    if training_arguments.local_rank in [0, -1]:  # Ensures only rank 0 or single-GPU saves
        model.config.save_pretrained(training_arguments.output_dir)
        model.save_pretrained(training_arguments.output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(training_arguments.output_dir, "non_lora_trainables.bin"))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    """Collects the state dict and dump to disk."""
    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
