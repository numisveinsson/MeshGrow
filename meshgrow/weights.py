"""Zenodo weight download and local cardiac weight staging."""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

from meshgrow.config import PipelineConfig, repo_root

logger = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/{record_id}"


@dataclass(frozen=True)
class WeightArchive:
    key: str
    record_id: Optional[str]
    zip_name: str
    extract_subdir: str
    marker_path: str


WEIGHT_ARCHIVES = {
    "seqseg": WeightArchive(
        key="seqseg",
        record_id="15020477",
        zip_name="nnUNet_results.zip",
        extract_subdir="seqseg",
        marker_path="seqseg/nnUNet_results",
    ),
    "linflonet": WeightArchive(
        key="linflonet",
        record_id="20802633",
        zip_name="LinFlo-Net_weights.zip",
        extract_subdir="linflonet",
        marker_path="linflonet/best_model.pth",
    ),
    "cardiac": WeightArchive(
        key="cardiac",
        record_id="20804513",
        zip_name="nnUNet_cardiac_weights.zip",
        extract_subdir="cardiac",
        marker_path="cardiac/nnUNet_cardiac_weights/Dataset015_HEARTTOTALSEGCT",
    ),
}


def _zenodo_download_url(record_id: str, zip_name: str) -> str:
    return (
        f"https://zenodo.org/api/records/{record_id}/files/"
        f"{zip_name}/content"
    )


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    with urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _relocate_extracted(root: Path, extract_subdir: str, zip_name: str) -> None:
    """Normalize common Zenodo zip layouts into expected subdirs."""
    target = root / extract_subdir
    target.mkdir(parents=True, exist_ok=True)

    if extract_subdir == "linflonet":
        candidates = list(root.rglob("best_model.pth"))
        if candidates:
            src = candidates[0]
            if src != target / "best_model.pth":
                shutil.copy2(src, target / "best_model.pth")
        return

    if extract_subdir == "seqseg":
        candidates = list(root.rglob("Dataset005_SEQAORTANDFEMOMR"))
        if candidates:
            nn_root = candidates[0].parent
            seqseg_root = target / "nnUNet_results"
            if nn_root != seqseg_root and nn_root.is_dir():
                if seqseg_root.exists():
                    shutil.rmtree(seqseg_root)
                shutil.move(str(nn_root), str(seqseg_root))
        return

    if extract_subdir == "cardiac":
        candidates = list(root.rglob("Dataset015_HEARTTOTALSEGCT"))
        if candidates:
            nn_root = candidates[0].parent
            cardiac_root = target / "nnUNet_cardiac_weights"
            if nn_root.name == "nnUNet_cardiac_weights":
                if cardiac_root.exists():
                    shutil.rmtree(cardiac_root)
                shutil.move(str(nn_root), str(cardiac_root))
            elif nn_root != cardiac_root:
                if cardiac_root.exists():
                    shutil.rmtree(cardiac_root)
                shutil.move(str(nn_root), str(cardiac_root))


def stage_local_cardiac_weights(dest: Path, source: Path) -> None:
    """Copy local cardiac nnU-Net weights into models/cardiac/."""
    target = dest / "cardiac" / "nnUNet_cardiac_weights"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    logger.info("Staged local cardiac weights from %s", source)


def download_weights(
    dest: Path,
    cfg: PipelineConfig,
    *,
    force: bool = False,
    cardiac_path: Optional[Path] = None,
) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    archives = {
        "seqseg": replace(
            WEIGHT_ARCHIVES["seqseg"], record_id=cfg.zenodo_seqseg_record
        ),
        "linflonet": replace(
            WEIGHT_ARCHIVES["linflonet"], record_id=cfg.zenodo_linflonet_record
        ),
        "cardiac": replace(
            WEIGHT_ARCHIVES["cardiac"], record_id=cfg.zenodo_cardiac_record
        ),
    }

    for archive in archives.values():
        marker = dest / archive.marker_path
        if marker.exists() and not force:
            logger.info("Skipping %s (already present)", archive.key)
            continue

        if archive.key == "cardiac":
            record_id = archive.record_id
            if record_id:
                zip_path = dest / "_downloads" / archive.zip_name
                url = _zenodo_download_url(record_id, archive.zip_name)
                download_file(url, zip_path)
                tmp = dest / "_downloads" / archive.key
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                extract_zip(zip_path, tmp)
                _relocate_extracted(tmp, archive.extract_subdir, archive.zip_name)
                _relocate_extracted(dest, archive.extract_subdir, archive.zip_name)
            else:
                local = cardiac_path or cfg.cardiac_weights_override
                if local is None:
                    local_candidate = repo_root() / "nnUNet_cardiac_weights"
                    if local_candidate.is_dir():
                        local = local_candidate
                if local is None or not Path(local).is_dir():
                    logger.warning(
                        "Cardiac Zenodo record not set and no local weights found. "
                        "Set zenodo.cardiac_record or pass --cardiac-path."
                    )
                    continue
                stage_local_cardiac_weights(dest, Path(local))
            continue

        if not archive.record_id:
            continue

        zip_path = dest / "_downloads" / archive.zip_name
        url = _zenodo_download_url(archive.record_id, archive.zip_name)
        download_file(url, zip_path)
        tmp = dest / "_downloads" / archive.key
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        extract_zip(zip_path, tmp)
        _relocate_extracted(tmp, archive.extract_subdir, archive.zip_name)
        _relocate_extracted(dest, archive.extract_subdir, archive.zip_name)


def _checkpoint_filename(name: str) -> str:
    return name if name.endswith(".pth") else f"{name}.pth"


def verify_weights(cfg: PipelineConfig) -> list[str]:
    """Return list of missing weight paths."""
    missing: list[str] = []
    checks = [
        ("CT cardiac nnU-Net", cfg.models_ct_nnunet_cardiac / "fold_all" / _checkpoint_filename(cfg.nnunet_checkpoint)),
        ("MR cardiac nnU-Net", cfg.models_mr_nnunet_cardiac / "fold_all" / _checkpoint_filename(cfg.nnunet_checkpoint)),
        ("LinFlo-Net", cfg.linflonet),
        ("SeqSeg nnU-Net root", cfg.seqseg_nnunet_root / "Dataset005_SEQAORTANDFEMOMR"),
    ]
    for label, path in checks:
        if not path.exists():
            missing.append(f"{label}: {path}")
    return missing
