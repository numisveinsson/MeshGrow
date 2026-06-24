"""Step 3: LinFlo-Net whole-heart mesh prediction."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from meshgrow.config import PipelineConfig
from meshgrow.io.staging import CasePaths

logger = logging.getLogger(__name__)


def run_linflonet_step(
    paths: CasePaths,
    cfg: PipelineConfig,
    modality: str,
    *,
    dry_run: bool = False,
) -> Path:
    if paths.is_step_complete("linflonet"):
        logger.info("LinFlo-Net step already complete for %s", paths.case_id)
        return paths.cardiac_mesh

    out_dir = paths.linflonet_dir
    cmd = [
        "linflonet",
        "predict",
        "--image",
        str(paths.subvolume),
        "--model",
        str(cfg.linflonet),
        "--modality",
        modality.lower(),
        "--output",
        str(out_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return paths.cardiac_mesh

    subprocess.run(cmd, check=True)

    meshes_dir = out_dir / "meshes"
    if not meshes_dir.is_dir():
        raise FileNotFoundError(f"LinFlo-Net meshes dir missing: {meshes_dir}")

    vtp_files = list(meshes_dir.glob("*.vtp"))
    if not vtp_files:
        raise FileNotFoundError(f"No .vtp meshes in {meshes_dir}")

    src = vtp_files[0]
    if len(vtp_files) > 1:
        for candidate in vtp_files:
            if paths.case_id in candidate.stem:
                src = candidate
                break

    target = paths.cardiac_mesh
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    return target
