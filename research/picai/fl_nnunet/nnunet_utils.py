import warnings
from logging import ERROR
from typing import Any, Dict, List, Tuple, Union

import torch
from flwr.common.logger import log
from numpy import ceil
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from research.picai.utils import MultiAttributeEnum

with warnings.catch_warnings():
    # silences a bunch of deprecation warnings related to scipy.ndimage
    # Raised an issue with nnunet. https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
    from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


# A MultiAttributeEnum Class to store nnunet config information
class NnUNetConfig(MultiAttributeEnum):
    # Define Member Attributes
    config_string: str
    default_num_processes: int
    num_spatial_dims: int

    # Override method so that we can define members with lists instead of dicts
    def get_attribute_keys(self, attributes: List) -> List[str]:
        return ["config_string", "default_num_processes", "num_spatial_dims"]

    # Define nnunet configs. Default for .value will be first item in list
    _2D = ["2d", 8, 2]
    _3D_FULLRES = ["3d_fullres", 4, 3]
    _3D_CASCADE = ["3d_cascade_fullres", 4, 3]
    _3D_LOWRES = ["3d_lowres", 8, 3]


def get_valid_nnunet_config(val: str) -> NnUNetConfig:
    try:
        return NnUNetConfig(val)
    except Exception as e:
        log(ERROR, f"Checking the nnunet configuration threw an exception {e}")
        raise e


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


class nnUNetDataLoaderWrapper(DataLoader):
    def __init__(
        self,
        nnunet_augmenter: Union[SingleThreadedAugmenter, NonDetMultiThreadedAugmenter, MultiThreadedAugmenter],
        nnunet_config: NnUNetConfig,
        infinite: bool = False,
    ) -> None:
        """
        Wraps nnunet dataloader classes using the pytorch dataloader to make
        them pytorch compatible. Also handles some unique stuff specific to
        nnunet such as deep supervision and infinite dataloaders

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
        self.num_spatial_dims = nnunet_config.num_spatial_dims

        # nnUNetDataloaders store their datasets under the self.data attribute
        self.dataset: nnUNetDataset = self.nnunet_dataloader._data
        super().__init__(dataset=self.dataset, batch_size=self.nnunet_dataloader.batch_size)

        # nnunet dataloaders are infinite by default so we have to track steps to stop iteration
        self.current_step = 0
        self.infinite = infinite
        self.len = len(self.dataset)

    def __next__(self) -> Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.infinite and self.current_step == self.__len__():
            self.reset()
            raise StopIteration
        else:
            self.current_step += 1
            batch = next(self.nnunet_augmenter)  # This returns a dictionary
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
        batch_size = self.nnunet_dataloader.batch_size
        return int(ceil(num_samples / batch_size))

    def reset(self) -> None:
        self.current_step = 0

    def __iter__(self) -> DataLoader:  # type: ignore
        # mypy gets angry that the return type is different
        return self

    def shutdown(self) -> None:
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
