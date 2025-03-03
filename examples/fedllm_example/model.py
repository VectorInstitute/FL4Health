import math
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    total_round: int,
    current_round: int = 0,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """
    Implement cosine annealing learning rate schedule for different local rounds.

    Args:
        total_round (int): The total number of rounds.
        current_round (int, optional): The current round. Defaults to 0.
        lrate_max (float, optional): The maximum learning rate. Defaults to 0.001.
        lrate_min (float, optional): The minimum learning rate. Defaults to 0.0.

    Returns:
        float: The learning rate for the current round.
    """

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """
    Load model with appropriate quantization config and other optimizations.
    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py

    Args:
        model_cfg (dict[str, Any]): The model configuration.

    Returns:
        torch.nn.Module: The model.
    """

    quantization_config = model_cfg["quantization"]
    if quantization_config == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization_config == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Use 4-bit or 8-bit quantization. You passed: {quantization_config}/")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model_cfg["gradient_checkpointing"])

    lora_config = model_cfg["lora"]
    assert isinstance(lora_config, dict)
    peft_config = LoraConfig(
        r=lora_config["peft_lora_r"],
        lora_alpha=lora_config["peft_lora_alpha"],
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)
