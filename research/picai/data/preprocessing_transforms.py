import numpy as np
import SimpleITK as sitk


def resample_img(
    image: sitk.Image,
    spacing: tuple[float, float, float],
    size: tuple[int, int, int] | None = None,
    is_label: bool = False,
    pad_value: float | int | None = 0.0,
) -> sitk.Image:
    """
    Resample images to target resolution spacing.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        spacing (tuple[float, float, float]): Target spacing between voxels in mm.
            Expected to be in Depth x Height x Width format.
        size (tuple[int, int, int]): Target size in voxels.
            Expected to be in Depth x Height x Width format.
        is_label (bool): Whether or not this is an annotation.
        pad_value (float | int | None): Amount of padding to use.

    Returns:
        (sitk.Image): The resampled image.
    """
    # get original spacing and size (x, y, z convention)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert PICAI z, y, x (Depth x Height x Width) convention to SimpleITK's convention x, y, z
    out_spacing = (spacing[2], spacing[1], spacing[0])

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
    return resample.Execute(image)


def input_verification_crop_or_pad(
    image: sitk.Image,
    size: tuple[int, int, int] = (20, 256, 256),
    physical_size: tuple[float, float, float] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Calculate target size for cropping and/or padding input image.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (tuple[int, int, int]): Target size in voxels.
            Expected to be in Depth x Height x Width format.
        physical_size (tuple[float, float, float]): Target size in mm. (Number of Voxels x Spacing)
            Expected to be in Depth x Height x Width format.

    Returns:
        (tuple[tuple[int, int, int], tuple[int, int, int]]):
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
        # verify size
        elif list(size) != list(size_zyx):
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
    assert rank <= len(shape) <= rank + 1, (
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"
    )

    return shape, size


def crop_or_pad(
    image: sitk.Image,
    size: tuple[int, int, int],
    physical_size: tuple[float, float, float] | None = None,
    crop_only: bool = False,
) -> sitk.Image:
    """
    Resize image by cropping and/or padding.

    Adapted from https://github.com/DIAGNijmegen/picai_prep.

    Args:
        image (sitk.Image): Image to be resized.
        size (tuple[int, int, int]): Target size in voxels. Expected to be in Depth x Height x Width format.
        physical_size (tuple[float, float, float] | None, optional): Target size in mm. (Number of Voxels x Spacing)
            Expected to be in  Depth x Height x Width format. Defaults to None.
        crop_only (bool, optional): Whether to only crop the image (True) or also perform padding (False). Defaults
            to False.

    Returns:
        (sitk.Image): Cropped or padded image.
    """
    # input conversion and verification
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # Since the subarrays are being set in the below for loop
    # We have to ensure that they are separate lists
    # and not the same reference (ie [[0, 0]] * rank)
    padding = [[0, 0] for _ in range(len(size))]
    slicer = [slice(None) for _ in range(len(size))]

    # for each dimension, determine process (cropping or padding)
    for i in range(len(size)):
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

    new_image = image[tuple(slicer)]

    # crop and/or pad image
    if any(pad[0] > 0 or pad[1] > 0 for pad in padding):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        new_image = pad_filter.Execute(new_image)
    return new_image
