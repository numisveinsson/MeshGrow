"""Image format conversion (NIfTI/MHA <-> VTI)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk

from meshgrow.io.resample import resample_image, transform_func
from meshgrow.vtk import minimal as vf

logger = logging.getLogger(__name__)


def change_sitk_vti(sitk_im: sitk.Image, *, label: bool = False) -> vtk.vtkImageData:
    """Convert SimpleITK to VTI image data (change_img_format.change_mha_vti)."""
    if label:
        sitk_im = sitk.Cast(sitk_im, sitk.sitkUInt8)
    vtk_im, _ = vf.export_sitk2vtk(sitk_im)
    return vtk_im


def images_share_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    if a.GetSize() != b.GetSize():
        return False
    if not np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0, atol=1e-6):
        return False
    if not np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0, atol=1e-6):
        return False
    return np.allclose(
        np.array(a.GetDirection()),
        np.array(b.GetDirection()),
        rtol=0,
        atol=1e-6,
    )


def resample_label_to_reference(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Align a label volume to the reference image grid."""
    if images_share_geometry(reference, label):
        return label

    same_frame = (
        np.allclose(label.GetOrigin(), reference.GetOrigin(), rtol=0, atol=1e-6)
        and np.allclose(
            np.array(label.GetDirection()),
            np.array(reference.GetDirection()),
            rtol=0,
            atol=1e-6,
        )
    )
    if same_frame:
        resampled = resample_image(
            label,
            target_size=[int(s) for s in reference.GetSize()],
            order=0,
        )
        if images_share_geometry(reference, resampled):
            return resampled
        logger.debug(
            "Resample-to-size produced grid %s; falling back to reference resample",
            resampled.GetSize(),
        )

    return transform_func(label, reference, sitk.Transform(), order=0)


def write_label_vti_to_reference(
    label_path: Path,
    reference: Path | sitk.Image,
    output_vti_path: Path,
) -> None:
    """Write a label volume as VTI on the reference image grid."""
    reference_im = (
        sitk.ReadImage(str(reference)) if isinstance(reference, Path) else reference
    )
    label = sitk.ReadImage(str(label_path))
    if not images_share_geometry(reference_im, label):
        logger.info(
            "Aligning vascular segmentation grid %s (spacing %s) to reference grid %s (spacing %s)",
            label.GetSize(),
            label.GetSpacing(),
            reference_im.GetSize(),
            reference_im.GetSpacing(),
        )
        label = resample_label_to_reference(label, reference_im)
    write_sitk_as_vti(label, output_vti_path, label=True)


def write_sitk_as_vti(
    sitk_im: sitk.Image,
    output_vti_path: Path,
    *,
    label: bool = False,
) -> None:
    vtk_im = change_sitk_vti(sitk_im, label=label)
    output_vti_path.parent.mkdir(parents=True, exist_ok=True)
    vf.write_img(str(output_vti_path), vtk_im)
