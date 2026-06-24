"""Step 4: SeqSeg vascular tracing with cardiac mesh seeds."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from meshgrow.config import PipelineConfig
from meshgrow.io.staging import CasePaths

logger = logging.getLogger(__name__)


def _write_seeds_json(data_dir: Path, case_id: str) -> None:
    seeds = [{"name": case_id, "seeds": [], "cardiac_mesh": True}]
    (data_dir / "seeds.json").write_text(
        json.dumps(seeds, indent=2) + "\n", encoding="utf-8"
    )


def stage_seqseg_data_dir(
    paths: CasePaths,
    cfg: PipelineConfig,
    input_image: Path,
) -> Path:
    data_dir = paths.seqseg_data_dir
    images_dir = data_dir / "images"
    meshes_dir = data_dir / "cardiac_meshes"
    images_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)

    target_image = images_dir / f"{paths.case_id}{cfg.seqseg_img_ext}"
    if input_image.resolve() != target_image.resolve():
        shutil.copy2(input_image, target_image)
    shutil.copy2(paths.cardiac_mesh, meshes_dir / f"{paths.case_id}.vtp")
    _write_seeds_json(data_dir, paths.case_id)
    return data_dir


def run_seqseg_step(
    paths: CasePaths,
    input_image: Path,
    cfg: PipelineConfig,
    modality: str,
    *,
    dry_run: bool = False,
) -> Path:
    if paths.is_step_complete("seqseg"):
        seg = paths.find_vascular_seg_mha()
        if seg is not None:
            logger.info("SeqSeg step already complete for %s", paths.case_id)
            return seg

    cmd = [
        "seqseg",
        "run",
        "batch",
        "-data_dir",
        str(paths.seqseg_data_dir),
        "-nnunet_results_path",
        str(cfg.seqseg_nnunet_root),
        "-train_dataset",
        cfg.seqseg_train_dataset(modality),
        "-img_ext",
        cfg.seqseg_img_ext,
        "-unit",
        cfg.seqseg_unit,
        "-scale",
        str(cfg.seqseg_scale),
        "-config_name",
        cfg.seqseg_config_name(modality),
        "-outdir",
        str(paths.seqseg_results),
        "-max_n_steps",
        str(cfg.seqseg_max_n_steps),
    ]
    if dry_run:
        logger.info("[dry-run] Would run: %s", " ".join(cmd))
        return paths.seqseg_results / f"{paths.case_id}_segmentation.mha"

    data_dir = stage_seqseg_data_dir(paths, cfg, input_image)
    paths.seqseg_results.mkdir(parents=True, exist_ok=True)
    cmd[4] = str(data_dir)

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    seg = paths.find_vascular_seg_mha()
    if seg is None:
        raise FileNotFoundError(
            f"SeqSeg segmentation not found in {paths.seqseg_results}"
        )
    return seg
