import io
import signal
import warnings
from collections.abc import Callable
from enum import Enum
from logging import WARNING, Logger
from typing import Any

import numpy as np
import torch
from flwr.common.logger import log
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from fl4health.utils.typing import LogLevel


with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet. https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
    from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

log(
    WARNING,
    (
        "You are using the old version of nnunet_utils from the research folder. "
        "Use the one from fl4health.utils instead"
    ),
)


class NnunetConfig(Enum):
    """
    The possible nnunet model configs as of nnunetv2 version 2.5.1.
    See https://github.com/MIC-DKFZ/nnUNet/tree/v2.5.1.
    """

    _2D = "2d"
    _3D_FULLRES = "3d_fullres"
    _3D_CASCADE = "3d_cascade_fullres"
    _3D_LOWRES = "3d_lowres"


NNUNET_DEFAULT_NP = {  # Nnunet's default number of processes for each config
    NnunetConfig._2D: 8,
    NnunetConfig._3D_FULLRES: 4,
    NnunetConfig._3D_CASCADE: 4,
    NnunetConfig._3D_LOWRES: 8,
}

NNUNET_N_SPATIAL_DIMS = {  # The number of spatial dims for each config
    NnunetConfig._2D: 2,
    NnunetConfig._3D_FULLRES: 3,
    NnunetConfig._3D_CASCADE: 3,
    NnunetConfig._3D_LOWRES: 3,
}


def use_default_signal_handlers(fn: Callable) -> Callable:
    """
    This is a decorator that resets the SIGINT and SIGTERM signal handlers back to the python defaults for the
    execution of the method.

    flwr 1.9.0 overrides the default signal handlers with handlers that raise an error on any interruption or
    termination. Since nnunet spawns child processes which inherit these handlers, when those subprocesses are
    terminated (which is expected behaviour), the flwr signal handlers raise an error (which we don't want).

    Flwr is expected to fix this in the next release. See the following issue:
    https://github.com/adap/flower/issues/3837
    """

    def new_fn(*args: Any, **kwargs: Any) -> Any:
        # Set SIGINT and SIGTERM back to defaults. Method returns previous handler
        sigint_old = signal.signal(signal.SIGINT, signal.default_int_handler)
        sigterm_old = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        # Execute function
        output = fn(*args, **kwargs)
        # Reset handlers back to what they were before function call
        signal.signal(signal.SIGINT, sigint_old)
        signal.signal(signal.SIGTERM, sigterm_old)
        return output

    return new_fn


# The two convert deepsupervision methods are necessary because fl4health requires
# predictions, targets and inputs to be single torch.Tensors or Dicts of torch.Tensors
def convert_deepsupervision_list_to_dict(
    tensor_list: list[torch.Tensor] | tuple[torch.Tensor], num_spatial_dims: int
) -> dict[str, torch.Tensor]:
    """
    Converts a list of torch.Tensors to a dictionary. Names the keys for each tensor based on the spatial resolution
    of the tensor and its index in the list. Useful for nnUNet models with deep supervision where model outputs and
    targets loaded by the dataloader are lists. Assumes the spatial dimensions of the tensors are last.

    Args:
        tensor_list (list[torch.Tensor]): A list of tensors, usually either nnunet model outputs or targets, to be
            converted into a dictionary.
        num_spatial_dims (int): The number of spatial dimensions. Assumes the spatial dimensions are last.

    Returns:
        (dict[str, torch.Tensor]): A dictionary containing the tensors as values where the keys are 'i-XxYxZ' where i
        was the tensor's index in the list and X,Y,Z are the spatial dimensions of the tensor.
    """
    # Convert list of targets into a dictionary
    tensors = {}

    for i, tensor in enumerate(tensor_list):
        # generate a key based on the spatial dimension and index
        key = str(i) + "-" + "x".join([str(s) for s in tensor.shape[-num_spatial_dims:]])
        tensors[key] = tensor

    return tensors


def convert_deepsupervision_dict_to_list(tensor_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """
    Converts a dictionary of tensors back into a list so that it can be used by nnunet deep supervision loss functions.

    Args:
        tensor_dict (dict[str, torch.Tensor]): Dictionary containing ``torch.Tensors``. The key values must start
            with 'X-' where X is an integer representing the index at which the tensor should be placed in the output
            list.

    Returns:
        (list[torch.Tensor]): A list of ``torch.Tensors``.
    """
    sorted_list = sorted(tensor_dict.items(), key=lambda x: int(x[0].split("-")[0]))
    return [tensor for key, tensor in sorted_list]


class NnUNetDataLoaderWrapper(DataLoader):
    def __init__(
        self,
        nnunet_augmenter: SingleThreadedAugmenter | NonDetMultiThreadedAugmenter | MultiThreadedAugmenter,
        nnunet_config: NnunetConfig | str,
        infinite: bool = False,
    ) -> None:
        """
        Wraps nnunet dataloader classes using the pytorch dataloader to make them pytorch compatible. Also handles
        some unique stuff specific to nnunet such as deep supervision and infinite dataloaders. The nnunet dataloaders
        should only be used for training and validation, not final testing.

        Args:
            nnunet_augmenter (SingleThreadedAugmenter | NonDetMultiThreadedAugmenter | MultiThreadedAugmenter): The
                dataloader used by nnunet.
            nnunet_config (NnunetConfig | str): The nnunet config. Enum type helps ensure that nnunet config is valid.
            infinite (bool, optional): Whether or not to treat the dataset as infinite. The dataloaders sample data
                with replacement either way. The only difference is that if set to False, a ``StopIteration`` is
                generated after ``num_samples``/``batch_size`` steps. Defaults to False.
        """
        # The augmenter is a wrapper on the nnunet dataloader
        self.nnunet_augmenter = nnunet_augmenter

        if isinstance(self.nnunet_augmenter, SingleThreadedAugmenter):
            self.nnunet_dataloader = self.nnunet_augmenter.data_loader
        else:
            self.nnunet_dataloader = self.nnunet_augmenter.generator

        # Figure out if dataloader is 2d or 3d
        self.num_spatial_dims = NNUNET_N_SPATIAL_DIMS[NnunetConfig(nnunet_config)]

        # nnUNetDataloaders store their datasets under the self.data attribute
        self.dataset: nnUNetDataset = self.nnunet_dataloader._data
        super().__init__(dataset=self.dataset, batch_size=self.nnunet_dataloader.batch_size)

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        if not self.infinite and self.current_step == self.__len__():
            self.reset()
            raise StopIteration  # Raise stop iteration after epoch has completed
        self.current_step += 1
        batch = next(self.nnunet_augmenter)  # This returns a dictionary
        # Note: When deep supervision is on, target is a list of segmentations at various scales
        # nnUNet has a wrapper for loss functions to enable deep supervision
        inputs: torch.Tensor = batch["data"]
        targets: torch.Tensor | list[torch.Tensor] = batch["target"]
        if isinstance(targets, list):
            target_dict = convert_deepsupervision_list_to_dict(targets, self.num_spatial_dims)
            return inputs, target_dict
        if isinstance(targets, torch.Tensor):
            return inputs, targets
        raise TypeError("Was expecting the target generated by the nnunet dataloader to be a list or a torch.Tensor")

    def __len__(self) -> int:
        """
        nnunetv2 v2.5.1 hardcodes an 'epoch' as 250 steps. We could set the len to
        n_samples/batch_size, but this gets complicated as nnunet models operate on
        patches of the input images, and therefore can have batch sizes larger than the
        dataset. We would then have epochs with only 1 step!

        Here we go through the hassle of computing the ratio between the number of
        voxels in a sample and the number of voxels in a patch and then using that
        factor to scale n_samples. This is particularly important for training 2d
        models on 3d data.
        """
        sample, _, _ = self.dataset.load_case(self.nnunet_dataloader.indices[0])
        n_image_voxels = np.prod(sample.shape)
        n_patch_voxels = np.prod(self.nnunet_dataloader.final_patch_size)
        # Scale factor is at least one to prevent shrinking the dataset. We can have a
        # larger patch size sometimes because nnunet will do padding
        scale = max(n_image_voxels / n_patch_voxels, 1)
        # Scale n_samples and then divide by batch size to get n_steps per epoch
        return round((len(self.dataset) * scale) / self.nnunet_dataloader.batch_size)

    def reset(self) -> None:
        self.current_step = 0

    def __iter__(self) -> DataLoader:  # type: ignore
        # mypy gets angry that the return type is different
        return self

    def shutdown(self) -> None:
        """The multithreaded augmenters used by nnunet need to be shutdown gracefully to avoid errors."""
        if isinstance(self.nnunet_augmenter, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            self.nnunet_augmenter._finish()
        else:
            del self.nnunet_augmenter


class Module2LossWrapper(_Loss):
    """Converts a ``nn.Module`` subclass to a ``_Loss`` subclass."""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class StreamToLogger(io.StringIO):
    def __init__(self, logger: Logger, level: LogLevel | int) -> None:
        """
        File-like stream object that redirects writes to a logger. Useful for redirecting stdout to a logger.

        Args:
            logger (Logger): The logger to redirect writes to.
            level (LogLevel): The log level at which to redirect the writes.
        """
        self.logger = logger
        self.level = level if isinstance(level, int) else level.value
        self.linebuf = ""  # idk why this is needed. Got this class from stack overflow

    def write(self, buf: str) -> int:
        char_count = 0
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
            char_count += len(line.rstrip())
        return char_count

    def flush(self) -> None:
        pass
