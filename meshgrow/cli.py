"""MeshGrow command-line interface."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from meshgrow import __version__
from meshgrow.config import bundled_example_config, load_config
from meshgrow.doctor import run_doctor
from meshgrow.pipeline.runner import run_pipeline
from meshgrow.weights import download_weights

logger = logging.getLogger("meshgrow")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meshgrow",
        description="Unified cardiac + vascular mesh pipeline",
    )
    parser.add_argument(
        "--version", action="version", version=f"meshgrow {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the pipeline")
    run_p.add_argument("--input", type=Path, help="Input images directory")
    run_p.add_argument("--output", type=Path, required=True, help="Output directory")
    run_p.add_argument(
        "--config", type=Path, default=None, help="Optional pipeline YAML"
    )
    run_p.add_argument(
        "--modality",
        choices=["ct", "mr"],
        default=None,
        help="Imaging modality",
    )
    run_p.add_argument("--case", action="append", dest="cases", help="Case ID(s)")
    run_p.add_argument(
        "--seqseg-max-steps",
        type=int,
        default=None,
        help="Maximum SeqSeg tracing steps (default: 200)",
    )
    run_p.add_argument("--from-step", choices=["nnunet", "crop", "linflonet", "seqseg", "combine"])
    run_p.add_argument("--to-step", choices=["nnunet", "crop", "linflonet", "seqseg", "combine"])
    run_p.add_argument("--dry-run", action="store_true")

    init_p = sub.add_parser("init", help="Scaffold project directory")
    init_p.add_argument(
        "--dest", type=Path, required=True, help="Project directory to create"
    )

    dl_p = sub.add_parser("download-weights", help="Download model weights from Zenodo")
    dl_p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Weights directory (default: MESHGROW_ROOT/models)",
    )
    dl_p.add_argument("--config", type=Path, default=None)
    dl_p.add_argument("--force", action="store_true")
    dl_p.add_argument(
        "--cardiac-path",
        type=Path,
        default=None,
        help="Local cardiac nnU-Net weights when Zenodo record is not set",
    )

    doc_p = sub.add_parser("doctor", help="Check dependencies and model weights")
    doc_p.add_argument("--config", type=Path, default=None)

    return parser


def cmd_init(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)
    target = dest / "pipeline.yaml"
    if not target.exists():
        shutil.copy2(bundled_example_config(), target)
    print(f"Initialized project at {dest}")
    print(f"  pipeline.yaml")
    print(f"  images/")
    return 0


def cmd_download_weights(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    dest = args.dest or cfg.weights_dir
    overrides = {}
    if args.cardiac_path:
        overrides["cardiac_weights_override"] = args.cardiac_path
    if overrides:
        cfg = load_config(args.config, overrides=overrides)
    download_weights(
        dest,
        cfg,
        force=args.force,
        cardiac_path=args.cardiac_path,
    )
    print(f"Weights downloaded to {dest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.input is None:
        print("--input is required for meshgrow run", file=sys.stderr)
        return 2

    overrides = {}
    if args.modality:
        overrides["modality"] = args.modality
    if args.seqseg_max_steps is not None:
        overrides.setdefault("seqseg", {})["max_n_steps"] = args.seqseg_max_steps
    cfg = load_config(args.config, overrides=overrides or None)
    modality = args.modality or cfg.modality

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_pipeline(
        input_dir=args.input.resolve(),
        output_dir=args.output.resolve(),
        cfg=cfg,
        modality=modality,
        case_ids=args.cases,
        from_step=args.from_step,
        to_step=args.to_step,
        dry_run=args.dry_run,
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    return run_doctor(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(args.dest)
    if args.command == "download-weights":
        return cmd_download_weights(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
