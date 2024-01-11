from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

import numpy as np
from typing import Dict, Optional, Sequence, Union

from research.picai.single_threaded_augmenter import SingleThreadedAugmenter
from research.picai.multi_threaded_augmenter import MultiThreadedAugmenter

default_3D_augmentation_params = {

    "do_elastic": False,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,
    "do_scaling": True,
    "scale_range": (0.7, 1.4),
    "independent_scale_factor_for_each_axis": False,
    "p_scale": 0.2,
    "do_rotation": True,
    "rotation_x": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_y": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_z": (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1, 2),
    "border_mode_data": "constant",
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
}


def apply_augmentations(dataloader: DataLoader, params: Dict = default_3D_augmentation_params, patch_size: Optional[int] = None, num_threads: int = 1,
                        border_val_seg: int = -1, seeds_train: Optional[Sequence[int]] = None, order_seg: int = 1, order_data: int = 3, disable: bool = False,
                        pin_memory: bool = False, use_multithreading: bool = True) -> Union[SingleThreadedAugmenter, MultiThreadedAugmenter]:
    """
    Apply augmentions to an already initialized batchgenerators DataLoader instance and return the resulting DataLoader.

    Args:
        dataloader (DataLoader): batchgenerator DataLoader.
        params (Dict[str, Any]): Dictionary of parameters that configure the transformations to be applied.
        patch_size (Optional[int]): TODO: Figure out what this is.
        num_threads (int): The number of threads to dedicated to loading data.
        border_val_seg (int): TODO: Figure out what this is.
        seeds_train (Optional[List[int]]): A list of seeds to sample from training.
        order_seg (int): TODO: Figure what this is.
        order_data (int): TODO: Figure what this is.
        disable (bool): TODO Figure what this is.
        pin_memory (bool): Whether or not to put fetched tensors into pinned memory (enables faster data transfer to CUDA-enabled GPUs).
        use_multithreading (book) Whether or not to use multithreading in data loading.

    Returns: 
       Union[SingleThreadedAugmenter, MultiThreadedAugmenter]: A batch generator generated from the DataLoader with the specified transformations applied.
    """
    # initialize list for train-time transforms
    tr_transforms = []
    if not disable:
        # morphological spatial transforms
        tr_transforms.append(SpatialTransform(patch_size,
                                              patch_center_dist_from_border=None,
                                              do_elastic_deform=params.get("do_elastic"),
                                              alpha=params.get("elastic_deform_alpha"),
                                              sigma=params.get("elastic_deform_sigma"),
                                              do_rotation=params.get("do_rotation"),
                                              angle_x=params.get("rotation_x"),
                                              angle_y=params.get("rotation_y"),
                                              angle_z=params.get("rotation_z"),
                                              p_rot_per_axis=params.get("rotation_p_per_axis"),
                                              do_scale=params.get("do_scaling"),
                                              scale=params.get("scale_range"),
                                              border_mode_data=params.get("border_mode_data"),
                                              border_cval_data=0,
                                              order_data=order_data,
                                              border_mode_seg="constant",
                                              border_cval_seg=border_val_seg,
                                              order_seg=order_seg,
                                              random_crop=params.get("random_crop"),
                                              p_el_per_sample=params.get("p_eldef"),
                                              p_scale_per_sample=params.get("p_scale"),
                                              p_rot_per_sample=params.get("p_rot"),
                                              independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")))

        # intensity transforms
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2, p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

        if params.get("do_additive_brightness"):
            tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                     params.get("additive_brightness_sigma"), True,
                                                     p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                     p_per_channel=params.get("additive_brightness_p_per_channel")))

        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                                            order_downsample=0,  order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=None))
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=0.1))  # inverted gamma

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))

        # flipping transform (reserved for last in order)
        if params.get("do_mirror") or params.get("mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    # convert from numpy to torch 
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    tr_transforms = Compose(tr_transforms)

    # Determine if using SingleThreadedAugmenter or MultThreadedAugmenter
    if use_multithreading and num_threads > 1:
        batchgenerator_train = MultiThreadedAugmenter(dataloader, tr_transforms, num_threads,
                                                          num_threads, seeds=seeds_train, 
                                                          pin_memory=pin_memory)
    else:
        batchgenerator_train = SingleThreadedAugmenter(dataloader, tr_transforms)
    return batchgenerator_train
