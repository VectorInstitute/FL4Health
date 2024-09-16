import io
import os
import signal
import sys
import warnings
from enum import Enum
from importlib import reload
from logging import DEBUG, INFO, Logger
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, no_type_check

import numpy as np
import torch
from flwr.common.logger import log
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
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


class NnunetConfig(Enum):
    """
    The possible nnunet model configs as of nnunetv2 version 2.5.1.
    See https://github.com/MIC-DKFZ/nnUNet/tree/v2.5.1
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
    This is a decorator that resets the SIGINT and SIGTERM signal handlers back to the
    python defaults for the execution of the method

    flwr 1.9.0 overrides the default signal handlers with handlers that raise an error
    on any interruption or termination. Since nnunet spawns child processes which
    inherit these handlers, when those subprocesses are terminated (which is expected
    behaviour), the flwr signal handlers raise an error (which we don't want).

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


def reload_modules(packages: Sequence[str]) -> None:
    """
    Given the names of one or more packages, subpackages or modules, reloads all the
    modules within the scope of each package or the modules themselves if a module was
    specified.

    Args:
        package (Sequence[str]): The absolute names of the packages, subpackages or
            modules to reload. The entire import hierarchy must be specified. Eg.
            'package.subpackage' to reload all modules in subpackage, not just
            'subpackage'. Packages are reloaded in the order they are given
    """
    for m_name, module in list(sys.modules.items()):
        for package in packages:
            if m_name.startswith(package):
                try:
                    reload(module)
                except Exception as e:
                    log(DEBUG, f"Failed to reload module {m_name}: {e}")


def set_nnunet_env(verbose: bool = False, **kwargs: str) -> None:
    """
    For each keyword argument name and value sets the current environment variable with
    the same name to that value and then reloads nnunet. Values must be strings. This
    is necessary because nnunet checks some environment variables on import, and
    therefore it must be imported or reloaded after they are set.
    """
    # Set environment variables
    for key, val in kwargs.items():
        os.environ[key] = val
        if verbose:
            log(INFO, f"Resetting env var '{key}' to '{val}'")

    # Its necessary to reload nnunetv2.paths first, then other modules with env vars
    reload_modules(["nnunetv2.paths"])
    reload_modules(["nnunetv2.default_n_proc_DA", "nnunetv2.configuration"])
    # Reload whatever depends on nnunetv2 environment variables
    # Be careful. If you reload something with an enum in it, things get messed up.
    reload_modules(["nnunetv2", "fl4health.clients.nnunet_client"])


# The two convert deepsupervision methods are necessary because fl4health requires
# predictions, targets and inputs to be single torch.Tensors or Dicts of torch.Tensors
def convert_deep_supervision_list_to_dict(
    tensor_list: Union[List[torch.Tensor], Tuple[torch.Tensor]], num_spatial_dims: int
) -> Dict[str, torch.Tensor]:
    """
    Converts a list of torch.Tensors to a dictionary. Names the keys for
    each tensor based on the spatial resolution of the tensor and its
    index in the list. Useful for nnUNet models with deep supervision where
    model outputs and targets loaded by the dataloader are lists. Assumes the
    spatial dimensions of the tensors are last.

    Args:
        tensor_list (List[torch.Tensor]): A list of tensors, usually either
            nnunet model outputs or targets, to be converted into a dictionary
        num_spatial_dims (int): The number of spatial dimensions. Assumes the
            spatial dimensions are last
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the tensors as
            values where the keys are 'i-XxYxZ' where i was the tensor's index
            in the list and X,Y,Z are the spatial dimensions of the tensor
    """
    # Convert list of targets into a dictionary
    tensors = {}

    for i, tensor in enumerate(tensor_list):
        # generate a key based on the spatial dimension and index
        key = str(i) + "-" + "x".join([str(s) for s in tensor.shape[-num_spatial_dims:]])
        tensors[key] = tensor

    return tensors


def convert_deep_supervision_dict_to_list(tensor_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a dictionary of tensors back into a list so that it can be used
    by nnunet deep supervision loss functions

    Args:
        tensor_dict (Dict[str, torch.Tensor]): Dictionary containing
            torch.Tensors. The key values must start with 'X-' where X is an
            integer representing the index at which the tensor should be placed
            in the output list

    Returns:
        List[torch.Tensor]: A list of torch.Tensors
    """
    sorted_list = sorted(tensor_dict.items(), key=lambda x: int(x[0].split("-")[0]))
    return [tensor for key, tensor in sorted_list]


def get_segs_from_probs(preds: torch.Tensor, has_regions: bool = False, threshold: float = 0.5) -> torch.Tensor:
    """
    Converts the nnunet model output probabilities to predicted segmentations

    Args:
        preds (torch.Tensor): The one hot encoded model output probabilities
            with shape (batch, classes, *additional_dims). The background should be a separate class
        has_regions (bool, optional): If True, predicted segmentations can be
            multiple classes at once. The exception is the background class
            which is assumed to be the first class (class 0). If False, each
            value in predicted segmentations has only a single class. Defaults to
            False.
        threshold (float): When has_regions is True, this is the threshold
            value used to determine whether or not an output is a part of a
            class

    Returns:
        torch.Tensor: tensor containing the predicted segmentations as a one hot encoded
            binary tensor of 64-bit integers.
    """
    if has_regions:
        pred_segs = preds > threshold
        # Mask is the inverse of the background class. Ensures that values
        # classified as background are not part of another class
        mask = ~pred_segs[:, 0]
        return pred_segs * mask
    else:
        pred_segs = preds.argmax(1)[:, None]  # shape (batch, 1, additional_dims)
        # one hot encode (OHE) predicted segmentations again
        # WARNING: Note the '_' after scatter. scatter_ and scatter are both
        # functions with different functionality. It is easy to introduce a bug
        # here by using the wrong one
        pred_segs_one_hot = torch.zeros(preds.shape, device=preds.device, dtype=torch.float32)
        pred_segs_one_hot.scatter_(1, pred_segs, 1)  # ohe -> One Hot Encoded
        # convert output preds to long since it is binary
        return pred_segs_one_hot.long()


def collapse_one_hot_tensor(input: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Collapses a one hot encoded tensor so that they are no longer one hot encoded.

    Args:
        input (torch.Tensor): The binary one hot encoded tensor

    Returns:
        torch.Tensor: Integer tensor with the specified dim collapsed
    """
    return torch.argmax(input.long(), dim=dim).to(input.device)


def get_dataset_n_voxels(source_plans: dict, n_cases: int) -> float:
    """
    Determines the total number of voxels in the dataset. Used by NnunetClient to
    determine the maximum batch size.

    Args:
        source_plans (Dict): The nnunet plans dict that is being modified
        n_cases (int): The number of cases in the dataset

    Returns:
        float: The total number of voxels in the local client dataset
    """
    # Need to determine input dimensionality
    if NnunetConfig._3D_FULLRES.value in source_plans["configurations"]:
        cfg = source_plans["configurations"][NnunetConfig._3D_FULLRES.value]
    else:
        cfg = source_plans["configurations"][NnunetConfig._2D.value]

    # Get total number of voxels in dataset
    image_shape = cfg["median_image_size_in_voxels"]
    approx_n_voxels = float(np.prod(image_shape, dtype=np.float64) * n_cases)
    return approx_n_voxels


def prepare_loss_arg(tensor: torch.Tensor | Dict[str, torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Converts pred and target tensors into the proper data type to be passed to the nnunet loss functions.

    Args:
        tensor (torch.Tensor | Dict[str, torch.Tensor]): The input tensor

    Returns:
        torch.Tensor | List[torch.Tensor]: The tensor ready to be passed to the loss
            function. A single tensor if not using deep supervision and a list of
            tensors if deep supervision is on.
    """
    # TODO: IDK why we have to make assumptions when we could just have a boolean state
    if isinstance(tensor, torch.Tensor):
        return tensor  # If input is a tensor then no changes required
    elif isinstance(tensor, dict):
        if len(tensor) > 1:  # Assume deep supervision is on and return a list
            return convert_deep_supervision_dict_to_list(tensor)
        else:  # If dict has only one item, assume deep supervision is off
            return list(tensor.values())[0]  # return the torch.Tensor


class nnUNetDataLoaderWrapper(DataLoader):
    def __init__(
        self,
        nnunet_augmenter: Union[SingleThreadedAugmenter, NonDetMultiThreadedAugmenter, MultiThreadedAugmenter],
        nnunet_config: Union[NnunetConfig, str],
        infinite: bool = False,
    ) -> None:
        """
        Wraps nnunet dataloader classes using the pytorch dataloader to make
        them pytorch compatible. Also handles some unique stuff specific to
        nnunet such as deep supervision and infinite dataloaders. The nnunet
        dataloaders should only be used for training and validation, not final testing.

        Args:
            nnunet_dataloader (Union[SingleThreadedAugmenter,
                NonDetMultiThreadedAugmenter]): The dataloader used by nnunet
            nnunet_config (NnUNetConfig): The nnunet config. Enum type helps
                ensure that nnunet config is valid
            infinite (bool, optional): Whether or not to treat the dataset
                as infinite. The dataloaders sample data with replacement
                either way. The only difference is that if set to False, a
                StopIteration is generated after num_samples/batch_size steps.
                Defaults to False.
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

    def __next__(self) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.infinite and self.current_step == self.__len__():
            self.reset()
            raise StopIteration  # Raise stop iteration after epoch has completed
        else:
            self.current_step += 1
            batch = next(self.nnunet_augmenter)  # This returns a dictionary
            # Note: When deep supervision is on, target is a list of ground truth
            # segmentations at various spatial scales/resolutions
            # nnUNet has a wrapper for loss functions to enable deep supervision
            inputs: torch.Tensor = batch["data"]
            targets: Union[torch.Tensor, List[torch.Tensor]] = batch["target"]
            if isinstance(targets, list):
                target_dict = convert_deep_supervision_list_to_dict(targets, self.num_spatial_dims)
                return inputs, target_dict
            elif isinstance(targets, torch.Tensor):
                return inputs, targets
            else:
                raise TypeError(
                    "Was expecting the target generated by the nnunet dataloader to be a list or a torch.Tensor"
                )

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
        """
        The multithreaded augmenters used by nnunet need to be shutdown gracefully to
        avoid errors
        """
        if isinstance(self.nnunet_augmenter, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            self.nnunet_augmenter._finish()
        else:
            del self.nnunet_augmenter


class Module2LossWrapper(_Loss):
    """Converts a nn.Module subclass to a _Loss subclass"""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class StreamToLogger(io.StringIO):
    def __init__(self, logger: Logger, level: Union[LogLevel, int]) -> None:
        """
        File-like stream object that redirects writes to a logger. Useful for redirecting stdout to a logger.

        Args:
            logger (Logger): The logger to redirect writes to
            level (LogLevel): The log level at which to redirect the writes
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


class PolyLRSchedulerWrapper(_LRScheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self._step_count: int
        super().__init__(optimizer, -1, False)

    # mypy incorrectly infers get_lr returns a float
    # Documented issue https://github.com/pytorch/pytorch/issues/100804
    @no_type_check
    def get_lr(self) -> Sequence[float]:
        curr_step = min(self._step_count, self.max_steps)
        new_lr = self.initial_lr * (1 - curr_step / self.max_steps) ** self.exponent
        return [new_lr] * len(self.optimizer.param_groups)
