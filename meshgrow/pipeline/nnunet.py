"""Step 1: nnU-Net cardiac binary segmentation."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from meshgrow.config import PipelineConfig, resolve_inference_device
from meshgrow.io.staging import CasePaths

logger = logging.getLogger(__name__)


def _nnunet_predict_cmd(model_folder: Path, cfg: PipelineConfig) -> list[str]:
    """Build nnU-Net predict command for the installed nnU-Net v2 CLI."""
    if shutil.which("nnUNetv2_predict_from_modelfolder"):
        return [
            "nnUNetv2_predict_from_modelfolder",
            "-m",
            str(model_folder),
            "-f",
            str(cfg.nnunet_fold),
            "-chk",
            cfg.nnunet_checkpoint,
        ]
    # Fall back to dataset-based predict (requires nnUNet_results env var).
    dataset_name = model_folder.parent.name
    return [
        "nnUNetv2_predict",
        "-d",
        dataset_name,
        "-c",
        cfg.nnunet_configuration,
        "-f",
        str(cfg.nnunet_fold),
        "-chk",
        cfg.nnunet_checkpoint,
    ]


def run_nnunet_step(
    paths: CasePaths,
    input_image: Path,
    cfg: PipelineConfig,
    modality: str,
    *,
    dry_run: bool = False,
) -> Path:
    if paths.is_step_complete("nnunet"):
        logger.info("nnU-Net step already complete for %s", paths.case_id)
        return paths.binary_seg

    model_folder = cfg.nnunet_cardiac_model(modality)
    paths.nnunet_input.mkdir(parents=True, exist_ok=True)
    paths.nnunet_output.mkdir(parents=True, exist_ok=True)

    staged_input = paths.nnunet_input / f"{paths.case_id}_0000.nii.gz"
    if not staged_input.exists() or staged_input.stat().st_mtime < input_image.stat().st_mtime:
        shutil.copy2(input_image, staged_input)

    cmd = _nnunet_predict_cmd(model_folder, cfg) + [
        "-i",
        str(paths.nnunet_input),
        "-o",
        str(paths.nnunet_output),
        "-device",
        resolve_inference_device(cfg.runtime_device),
    ]
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return paths.binary_seg

    subprocess.run(cmd, check=True)

    candidates = list(paths.nnunet_output.glob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No nnU-Net output in {paths.nnunet_output}")
    pred = candidates[0]
    paths.binary_seg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred, paths.binary_seg)
    logger.info("Wrote binary seg %s", paths.binary_seg)
    return paths.binary_seg
