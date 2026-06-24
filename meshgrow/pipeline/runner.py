"""Pipeline orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from meshgrow.config import PipelineConfig
from meshgrow.io.staging import (
    PIPELINE_STEPS,
    CasePaths,
    discover_cases,
    find_image_file,
    steps_from,
    steps_until,
)
from meshgrow.pipeline import combine, crop, linflonet, nnunet, seqseg

logger = logging.getLogger(__name__)


def run_case(
    paths: CasePaths,
    input_dir: Path,
    cfg: PipelineConfig,
    modality: str,
    *,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    input_image = find_image_file(input_dir, paths.case_id)
    selected_from = list(steps_from(from_step))
    selected_until = set(steps_until(to_step))
    steps = [s for s in selected_from if s in selected_until]

    for step in steps:
        logger.info("Case %s — step %s", paths.case_id, step)
        if step == "nnunet":
            nnunet.run_nnunet_step(
                paths, input_image, cfg, modality, dry_run=dry_run
            )
        elif step == "crop":
            crop.run_crop_step(paths, input_image, cfg, dry_run=dry_run)
        elif step == "linflonet":
            linflonet.run_linflonet_step(paths, cfg, modality, dry_run=dry_run)
        elif step == "seqseg":
            seqseg.run_seqseg_step(
                paths, input_image, cfg, modality, dry_run=dry_run
            )
        elif step == "combine":
            combine.run_combine_step(
                paths, input_image, cfg, dry_run=dry_run
            )

        if not dry_run:
            paths.mark_step_complete(step)


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    modality: str,
    case_ids: Optional[list[str]] = None,
    from_step: Optional[str] = None,
    to_step: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    _configure_file_logging(log_path)

    cases = case_ids or discover_cases(input_dir)
    logger.info("Processing %d case(s): %s", len(cases), ", ".join(cases))

    errors: list[str] = []
    for case_id in cases:
        paths = CasePaths(output_dir, case_id)
        try:
            run_case(
                paths,
                input_dir,
                cfg,
                modality,
                from_step=from_step,
                to_step=to_step,
                dry_run=dry_run,
            )
        except Exception as exc:
            msg = f"{case_id}: {exc}"
            logger.exception(msg)
            errors.append(msg)
            if not cfg.runtime_continue_on_error:
                raise

    if errors and cfg.runtime_continue_on_error:
        raise RuntimeError("One or more cases failed:\n" + "\n".join(errors))


def _configure_file_logging(log_path: Path) -> None:
    root = logging.getLogger("meshgrow")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)
