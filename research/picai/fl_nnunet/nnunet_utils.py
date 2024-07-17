import contextlib
import sys
import warnings
from typing import Any, Dict, Generator, List, Literal, Tuple, Union, get_args

import torch
from numpy import ceil
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet. https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

nnUNetConfig = Literal["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]


class DummyFile(object):
    def write(self, x: Any) -> None:
        pass


# Define a stdout silencer so that nnunet doesn't print a bunch of stuff and
# clutter the console
@contextlib.contextmanager
def nostdout() -> Generator[Any, Any, Any]:
    """
    Silences the stdout for any code ran within its context.

    Example usage:
        with nostdout():
            # code in here runs quietly
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()  # type: ignore
    yield
    sys.stdout = save_stdout


def get_num_spatial_dims(nnunet_config: str) -> int:
    """
    Returns the number of spatial dimensions given the nnunet configuration

    Args:
        nnunet_config (nnUNetConfig): string specifying the nnunet config

    Returns:
        int: The number of spatial dimensions
    """
    if nnunet_config == "2d":
        return 2
    elif nnunet_config in ["3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
        return 3
    else:
        raise TypeError(f"Got unexpected nnunet config: {nnunet_config}")


def convert_deepsupervision_list_to_dict(
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


def convert_deepsupervision_dict_to_list(tensor_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
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


def is_valid_config(val: str) -> bool:
    return val in list(get_args(nnUNetConfig))


class nnUNetDataLoaderWrapper(DataLoader):
    def __init__(self, nnunet_dataloader: nnUNetDataLoaderBase, nnunet_config: str, infinite: bool = True) -> None:
        """
        Wraps nnunet dataloader classes using the pytorch dataloader to make
        them pytorch compatible. Also handles some unique stuff specific to
        nnunet such as deep supervision and infinite dataloaders

        Args:
            nnunet_dataloader (nnUNetDataLoaderBase): The nnunet dataloader
            nnunet_config (str): The nnunet config
            infinite (bool, optional): Whether or not to treat the dataset
                as infinite. Defaults to True.
        """
        self.nnunet_dataloader = nnunet_dataloader

        # Figure out if dataloader is 2d or 3d
        self.num_spatial_dims = get_num_spatial_dims(nnunet_config)

        # nnUNetDataloaders store their datasets under the self.data attribute
        self.dataset: nnUNetDataset = self.nnunet_dataloader.generator._data
        super().__init__(dataset=self.dataset, batch_size=self.nnunet_dataloader.generator.batch_size)

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite
        self.len = len(self.dataset)

    def __next__(self) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.infinite and self.current_step == self.len:
            self.reset()
            raise StopIteration
        else:
            self.current_step += 1
            batch = next(self.nnunet_dataloader)  # This returns a dictionary
            # Note: When deep supervision is on, target is a list of segmentations at various scales
            # nnUNet has a wrapper for loss functions to enable deep supervision
            inputs: torch.Tensor = batch["data"]
            targets: Union[torch.Tensor, List[torch.Tensor]] = batch["target"]
            if isinstance(targets, list):
                target_dict = convert_deepsupervision_list_to_dict(targets, self.num_spatial_dims)
                return inputs, target_dict
            elif isinstance(targets, torch.Tensor):
                return inputs, targets
            else:
                raise TypeError(
                    "Was expecting the target generated by the nnunet dataloader to be a list or a torch.Tensor"
                )

    def __len__(self) -> int:
        """
        nnUNetDataloaders are 'infinite' meaning they randomly sample batches
        from the dataset. This makes the distinction between
        num_samples // batch_size and ceil(num_samples/batch_size)
        meaningless. We will abritrarily use the later

        Returns:
            int: integer equal to num_samples // batch_size
        """
        num_samples = len(self.dataset)
        batch_size = self.nnunet_dataloader.generator.batch_size
        return int(ceil(num_samples / batch_size))

    def reset(self) -> None:
        self.current_step = 0

    def __iter__(self) -> DataLoader:  # type: ignore
        # mypy gets angry that the return type is different
        return self


class Module2LossWrapper(_Loss):
    """Converts a nn.Module subclass to a _Loss subclass"""

    def __init__(self, loss: nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.loss = loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)
