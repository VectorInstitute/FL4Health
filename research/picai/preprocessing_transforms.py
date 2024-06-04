from typing import Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk


def resample_img(
    image: sitk.Image,
    spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    size: Optional[Tuple[int, int, int]] = None,
    is_label: bool = False,
    pad_value: Optional[Union[float, int]] = 0.0,
) -> sitk.Image:
    """
    Resample images to target resolution spacing.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        spacing (Tuple[float, float, float]): Target spacing between voxels in mm.
            Expected to be in Depth x Height x Width format.
        size (Tuple[int, int, int]): Target size in voxels.
            Expected to be in Depth x Height x Width format.
        is_label (bool): Whether or not this is an annotation.
        pad_value (Optional[Union[float, int]]): Amount of padding to use.

    Returns:
        sitk.Image: The resampled image.
    """
    # get original spacing and size (x, y, z convention)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert PICAI z, y, x (Depth x Height x Width) convention to SimpleITK's convention x, y, z
    out_spacing = list(reversed(spacing))

    if size is None:
        # calculate output size in voxels
        size = (
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
        )
    else:
        # If size is passed, we assume it is in z, y, x so we need to reverse.
        size = (size[2], size[1], size[0])

    # determine pad value
    if pad_value is None:
        # PixelIDValue is the default pixel value (which is used for padding)
        # Defaults to 0
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


def input_verification_crop_or_pad(
    image: sitk.Image,
    size: Tuple[int, int, int] = (20, 256, 256),
    physical_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Calculate target size for cropping and/or padding input image.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (Tuple[int, int, int]): Target size in voxels.
            Expected to be in Depth x Height x Width format.
        physical_size (Tuple[float, float, float]): Target size in mm. (Number of Voxels x Spacing)
            Expected to be in Depth x Height x Width format.

    Returns:
        Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
            Shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x)) and
            Size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        spacing_zyx = list(reversed(image.GetSpacing()))

        size_zyx = (
            int(np.round(physical_size[0] / spacing_zyx[0])),
            int(np.round(physical_size[1] / spacing_zyx[1])),
            int(np.round(physical_size[2] / spacing_zyx[2])),
        )

        if size is None:
            # use physical size
            size = size_zyx
        else:
            # verify size
            if list(size) != list(size_zyx):
                raise ValueError(
                    f"Size and physical size do not match. Size: {size}, physical size: "
                    f"{physical_size}, spacing: {spacing_zyx}, size_zyx: {size_zyx}."
                )

    if isinstance(image, sitk.Image):
        # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
        shape = image.GetSize()
        size = (size[2], size[1], size[0])
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
    rank = len(size)
    assert (
        rank <= len(shape) <= rank + 1
    ), f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

    return shape, size


def crop_or_pad(
    image: sitk.Image,
    size: Tuple[int, int, int] = (20, 256, 256),
    physical_size: Optional[Tuple[float, float, float]] = None,
    crop_only: bool = False,
) -> sitk.Image:
    """
    Resize image by cropping and/or padding.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (Tuple[int, int, int]): Target size in voxels.
            Expected to be in Depth x Height x Width format.
        physical_size (Tuple[float, float, float]): Target size in mm. (Number of Voxels x Spacing)
            Expected to be in Depth x Height x Width format.

    Returns:
        sitk.Image: Cropped or padded image.
    """
    # input conversion and verification
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    rank = len(size)
    padding = [[0, 0]] * rank
    slicer = [slice(None)] * rank

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if shape[i] < size[i]:
            if crop_only:
                continue

            # set padding settings
            padding[i][0] = (size[i] - shape[i]) // 2
            padding[i][1] = size[i] - shape[i] - padding[i][0]
        else:
            # create slicer object to crop image
            idx_start = int(np.floor((shape[i] - size[i]) / 2.0))
            idx_end = idx_start + size[i]
            slicer[i] = slice(idx_start, idx_end)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image[tuple(slicer)])
    else:
        return sitk.GetImageFromArray(np.pad(image[tuple(slicer)], padding))
