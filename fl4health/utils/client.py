import copy
import os
from collections.abc import Iterable
from inspect import currentframe, getframeinfo
from logging import INFO, LogRecord
from typing import Any, TypeVar

import torch
import torch.nn as nn
from flwr.common.logger import LOGGER_NAME, console_handler, log
from flwr.common.typing import Config, Scalar
from tqdm import tqdm

from fl4health.utils.config import narrow_dict_type
from fl4health.utils.logging import LoggingMode
from fl4health.utils.metrics import MetricPrefix
from fl4health.utils.typing import TorchInputType, TorchTargetType

T = TypeVar("T", TorchInputType, TorchTargetType)


def fold_loss_dict_into_metrics(
    metrics: dict[str, Scalar], loss_dict: dict[str, float], logging_mode: LoggingMode
) -> None:
    # Prefixing the loss value keys with the mode from which they are generated
    if logging_mode is LoggingMode.VALIDATION:
        metrics.update({f"{MetricPrefix.VAL_PREFIX.value} {key}": loss_val for key, loss_val in loss_dict.items()})
    else:
        metrics.update({f"{MetricPrefix.TEST_PREFIX.value} {key}": loss_val for key, loss_val in loss_dict.items()})


def set_pack_losses_with_val_metrics(config: Config) -> bool:
    try:
        pack_losses_with_val_metrics = narrow_dict_type(config, "pack_losses_with_val_metrics", bool)
    except ValueError:
        pack_losses_with_val_metrics = False
    if pack_losses_with_val_metrics:
        log(INFO, "As specified in the config, all validation losses will be packed into validation metrics")
    return pack_losses_with_val_metrics


def move_data_to_device(data: T, device: torch.device) -> T:
    """
    _summary_

    Args:
        data (T): The data to move to self.device. Can be a TorchInputType or a TorchTargetType
        device (torch.device): Device indicator for where to send the model, batches, labels etc. Often 'cpu' or
            'cuda'

    Raises:
        TypeError: Raised if data is not one of the types specified by TorchInputType or TorchTargetType

    Returns:
        T: The data argument except now it's been moved to self.device
    """
    # Currently we expect both inputs and targets to be either tensors
    # or dictionaries of tensors
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: value.to(device) for key, value in data.items()}
    else:
        raise TypeError(
            "data must be of type torch.Tensor or dict[str, torch.Tensor]. If definition of TorchInputType or "
            "TorchTargetType has changed this method might need to be updated or split into two."
        )


def check_if_batch_is_empty_and_verify_input(input: TorchInputType) -> bool:
    """
    This function checks whether the provided batch (input) is empty. If the input is a dictionary of inputs, it
    first verifies that the length of all inputs is the same, then checks if they are non-empty.
    NOTE: This function assumes the input is BATCH FIRST

    Args:
        input (TorchInputType): Input batch. input can be of type torch.Tensor or dict[str, torch.Tensor], and in the
        latter case, the batch is considered to be empty if all tensors in the dictionary have length zero.

    Raises:
        TypeError: Raised if input is not of type torch.Tensor or dict[str, torch.Tensor].
        ValueError: Raised if input has type dict[str, torch.Tensor] and not all tensors within the dictionary have
            the same size.

    Returns:
        bool: True if input is an empty batch.
    """
    if isinstance(input, torch.Tensor):
        return len(input) == 0
    elif isinstance(input, dict):
        input_iter = iter(input.items())
        _, first_val = next(input_iter)
        first_val_len = len(first_val)
        if not all(len(val) == first_val_len for _, val in input_iter):
            raise ValueError("Not all tensors in the dictionary have the same size.")
        else:
            return first_val_len == 0
    else:
        raise TypeError("Input must be of type torch.Tensor or dict[str, torch.Tensor].")


def clone_and_freeze_model(model: nn.Module) -> nn.Module:
    """
    Creates a clone of the model with frozen weights to be used in loss calculations so the original model is
    preserved in its current state.

    Args:
        model (nn.Module): Model to clone and freeze
    Returns:
        nn.Module: Cloned and frozen model
    """

    cloned_model = copy.deepcopy(model)
    for param in cloned_model.parameters():
        param.requires_grad = False
    cloned_model.eval()

    return cloned_model


def maybe_progress_bar(iterable: Iterable, display_progress_bar: bool) -> Iterable:
    """
    Used to print progress bars during client training and validation. If
    self.progress_bar is false, just returns the original input iterable
    without modifying it.
    Args:
        iterable (Iterable): The iterable to wrap
    Returns:
        Iterable: an iterator which acts exactly like the original
            iterable, but prints a dynamically updating progress bar every
            time a value is requested. Or the original iterable if
            self.progress_bar is False
    """
    if not display_progress_bar:
        return iterable
    else:
        # We can use the flwr console handler to format progress bar
        frame = currentframe()
        lineno = 0 if frame is None else getframeinfo(frame).lineno
        record = LogRecord(
            name=LOGGER_NAME,
            pathname=os.path.abspath(os.getcwd()),
            lineno=lineno,  #
            args={},
            exc_info=None,
            level=INFO,
            msg="{l_bar}{bar}{r_bar}",
        )
        format = console_handler.format(record)
        # Create a clean looking tqdm instance that matches the flwr logging
        kwargs: Any = {
            "leave": True,
            "ascii": " >=",
            "unit": "steps",
            "dynamic_ncols": True,
            "bar_format": format,
        }
        return tqdm(iterable, **kwargs)
