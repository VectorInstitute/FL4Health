import os
from collections.abc import Iterator
from logging import WARNING
from typing import Any

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from flwr.common.logger import log
from torch.nn import Parameter
from transformers import Trainer, TrainingArguments


def maybe_zero_3(param: Any, ignore_status: bool = False, name: str | None = None) -> Any:
    """
    If stage 3 ZeRo is enabled, gather the parameter and return the gathered parameter.

    Args:
        param (Any): The parameter to gather.
        ignore_status (bool, optional): Whether to ignore the status of the parameter. Defaults to False.
        name (str, optional): The name of the parameter. Defaults to None.

    Returns:
        (Any): The gathered parameter.
    """
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            log(WARNING, f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params: Iterator[tuple[str, Parameter]], bias: str) -> dict[str, Any]:
    """
    Get the state dict for the PEFT model when stage 3 ZeRo is enabled.

    Args:
        named_params (Iterator[tuple[str, Parameter]]): The named parameters of the model.
        bias (str): The bias to consider.

    Returns:
        (dict[str, Any]): The state dict for the PEFT model.
    """
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
        for t in maybe_lora_bias.values():
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    # We should gather all parameters in the model
    return {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}


def get_peft_state_non_lora_maybe_zero_3(
    named_params: Iterator[tuple[str, Parameter]], require_grad_only: bool = True
) -> dict[str, Any]:
    """
    Get the state dict for the non-LoRA trainable parameters when stage 3 ZeRo is enabled.

    Args:
        named_params (Iterator[tuple[str, Parameter]]): The named parameters of the model.
        require_grad_only (bool, optional): Whether to require gradients. Defaults to True.

    Returns:
        (dict[str, Any]): The state dict for the non-LoRA trainable parameters.
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    # We should gather all parameters in the model
    return {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}


def safe_save_model_for_zero3(model: torch.nn.Module, training_arguments: TrainingArguments) -> None:
    """
    Saves PEFT model and non-LoRA trainable parameters when stage 3 ZeRo is enabled.

    Args:
        model: The model to save.
        training_arguments: Training arguments containing local_rank and output directory.
    """
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), "none")
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())

    if training_arguments.local_rank in [0, -1]:  # Ensures only rank 0 or single-GPU saves
        # type ignore because the type in transformers is very indirect
        model.config.save_pretrained(training_arguments.output_dir)  # type: ignore
        # type ignore because the type in transformers is very indirect
        model.save_pretrained(training_arguments.output_dir, state_dict=state_dict)  # type: ignore
        assert training_arguments.output_dir is not None, "training_arguments.output_dir must not be None"
        torch.save(non_lora_state_dict, os.path.join(training_arguments.output_dir, "non_lora_trainables.bin"))


def safe_save_model_for_hf_trainer(trainer: Trainer) -> None:
    """
    Safely save the model for the Hugging Face Trainer. This module waits for all processes to synchronize
    before saving the model.

    Args:
        trainer (transformers.Trainer): The trainer.
    """
    trainer.accelerator.wait_for_everyone()  # type:ignore
    torch.cuda.synchronize()

    trainer.save_model()  # type:ignore
