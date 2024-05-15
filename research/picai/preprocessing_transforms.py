from typing import Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk


def resample_img(
    image: sitk.Image,
    spacing: Sequence[float] = (2.0, 2.0, 2.0),
    size: Optional[Sequence[int]] = None,
    is_label: bool = False,
    pad_value: Optional[Union[float, int]] = 0.0,
) -> sitk.Image:
    """
    Resample images to target resolution spacing.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        spacing (Sequence[float]): Spacing between voxels along each dimentsion.
        size (Sequence[int]): Target size in voxels (z, y, x).
        is_label (bool): Whether or not this is an annotation.
        pad_value (Optional[Union[float, int]]): Amount of padding to use.

    Returns:
        sitk.Image: The resampled image.
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert PICAI z, y, x convention to SimpleITK's convention
    out_spacing = list(spacing)[::-1]

    if size is None:
        # calculate output size in voxels
        size = [
            int(np.round(size * (spacing_in / spacing_out)))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
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
    size: Sequence[int] = (20, 256, 256),
    physical_size: Optional[Sequence[float]] = None,
) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Calculate target size for cropping and/or padding input image.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (Sequence[int]): Target size in voxels (z, y, x).
        physical_size: Target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
        Tuple[Sequence[int], Sequence[int]]:
            Shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x)) and
            Size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        spacing_zyx = list(image.GetSpacing())[::-1]
        size_zyx = [length / spacing for length, spacing in zip(physical_size, spacing_zyx)]
        size_zyx = [int(np.round(x)) for x in size_zyx]

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
        size = list(size)[::-1]
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size)
    rank = len(size)
    assert (
        rank <= len(shape) <= rank + 1
    ), f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

    return shape, size


def crop_or_pad(
    image: sitk.Image,
    size: Sequence[int] = (20, 256, 256),
    physical_size: Optional[Sequence[float]] = None,
    crop_only: bool = False,
) -> sitk.Image:
    """
    Resize image by cropping and/or padding.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (Sequence[int]): Target size in voxels (z, y, x).
        physical_size: Target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
        sitk.Image: Cropped or padded image.
    """
    # input conversion and verification
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    rank = len(size)
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

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
