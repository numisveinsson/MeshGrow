"""Step 2: crop subvolume around cardiac binary segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from meshgrow.config import PipelineConfig
from meshgrow.io.staging import CasePaths

logger = logging.getLogger(__name__)


def extract_subvolume(
    image_path: Path,
    binary_seg_path: Path,
    output_path: Path,
    padding_voxels: int = 20,
) -> Path:
    img = sitk.ReadImage(str(image_path))
    binary_seg = sitk.ReadImage(str(binary_seg_path))

    img_reader = sitk.ImageFileReader()
    img_reader.SetFileName(str(image_path))
    img_reader.LoadPrivateTagsOn()
    img_reader.ReadImageInformation()

    binary_np = sitk.GetArrayFromImage(binary_seg).transpose(2, 1, 0)
    locs = np.where(binary_np == 1)
    if locs[0].size == 0:
        raise ValueError(f"No foreground voxels in binary seg {binary_seg_path}")

    img_size = np.array(img.GetSize(), dtype=int)
    mins = np.array([locs[i].min() for i in range(3)], dtype=int)
    maxs = np.array([locs[i].max() for i in range(3)], dtype=int)

    subvolume_index = np.maximum(mins - padding_voxels, 0)
    subvolume_max = np.minimum(maxs + padding_voxels, img_size - 1)
    subvolume_size = (subvolume_max - subvolume_index + 1).astype(int).tolist()
    subvolume_index = subvolume_index.astype(int).tolist()

    img_reader.SetExtractSize(subvolume_size)
    img_reader.SetExtractIndex(subvolume_index)
    subvolume = img_reader.Execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(subvolume, str(output_path))
    logger.info(
        "Wrote subvolume %s (size=%s index=%s, padding=%d voxels)",
        output_path,
        subvolume_size,
        subvolume_index,
        padding_voxels,
    )
    return output_path


def run_crop_step(
    paths: CasePaths,
    input_image: Path,
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
) -> Path:
    if paths.is_step_complete("crop"):
        logger.info("Crop step already complete for %s", paths.case_id)
        return paths.subvolume

    if dry_run:
        logger.info("[dry-run] Would crop %s -> %s", input_image, paths.subvolume)
        return paths.subvolume

    return extract_subvolume(
        input_image,
        paths.binary_seg,
        paths.subvolume,
        cfg.crop_padding_voxels,
    )
