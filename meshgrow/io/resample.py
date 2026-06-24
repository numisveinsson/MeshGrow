"""Resampling helpers ported from vascular-segment-sampler."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def resample(
    sitk_im: sitk.Image | str,
    resolution: tuple[float, float, float] = (0.5, 0.5, 0.5),
    order: int = 1,
) -> sitk.Image:
    """Resample to target spacing while preserving origin, direction, and extent."""
    image = sitk.ReadImage(str(sitk_im)) if isinstance(sitk_im, str) else sitk_im
    resampler = sitk.ResampleImageFilter()
    if order == 1:
        resampler.SetInterpolator(sitk.sitkLinear)
    elif order == 2:
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(resolution)
    orig_size = np.array(image.GetSize(), dtype=np.int64)
    orig_spacing = np.array(image.GetSpacing())
    resolution_arr = np.array(resolution, dtype=np.float64)
    new_size = orig_size * (orig_spacing / resolution_arr)
    new_size = np.ceil(new_size).astype(np.int64)
    resampler.SetSize([int(s) for s in new_size])
    return resampler.Execute(image)


def resample_image(
    img_sitk: sitk.Image,
    *,
    target_size: list[int] | tuple[int, int, int] | None = None,
    target_spacing: list[float] | tuple[float, float, float] | None = None,
    order: int = 1,
) -> sitk.Image:
    """Resample to a target size or spacing (see change_img_resample.py)."""
    if target_size is None and target_spacing is None:
        raise ValueError("Either target_size or target_spacing must be provided")
    if target_size is not None:
        new_res = [
            img_sitk.GetSpacing()[i] * (img_sitk.GetSize()[i] / target_size[i])
            for i in range(3)
        ]
    else:
        new_res = target_spacing
    return resample(img_sitk, resolution=tuple(new_res), order=order)


def transform_func(
    image: sitk.Image,
    reference_image: sitk.Image,
    transform: sitk.Transform,
    order: int = 1,
) -> sitk.Image:
    """Resample ``image`` onto the grid of ``reference_image``."""
    if order == 1:
        interpolator = sitk.sitkLinear
    elif order == 0:
        interpolator = sitk.sitkNearestNeighbor
    elif order == 3:
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError(f"Unsupported interpolation order: {order}")
    return sitk.Resample(
        image,
        reference_image,
        transform,
        interpolator,
        0,
        image.GetPixelID(),
    )
